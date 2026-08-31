# Pack Your Bags — Project Write-up

## What it is

Pack Your Bags is a multi-agent trip-planning system. A traveler fills in one form — nationality, destination country and cities, dates, where they're flying from, purpose, a few preferences, and an email — and gets back a single grounded trip briefing: whether they need a visa and the deadlines that come with it, real flights and hotels, and a day-by-day plan. Under the hood an orchestrator coordinates three specialized agents, passes real state between them, recovers when a tool fails, and stops to ask a human before it writes anything to a calendar or sends an email.

It's a Week 3 project. The retrieval and live-API groundwork is carried over from a Week 2 RAG project; the new work here is the orchestration layer — the control flow, the shared state, the failure handling, the human-in-the-loop gates, and one tool exposed over MCP. The UI is Streamlit; the orchestrator is a LangGraph state graph; the reasoning steps run on Claude.

## The use case

Planning a trip means opening a dozen tabs. You check the destination government's visa page for your nationality. You compare flights on two or three aggregators. You look at hotels in each city you're visiting. You research restaurants and things to do. You try to remember that some visas need to be applied for weeks ahead, and you hope you don't miss a deadline. Pack Your Bags collapses that into one pass: you describe the trip once, and it does the checking, comparing and researching for you, then puts a day-by-day plan on top.

The bar I set for "it works" was deliberately not the happy path. It's this: a traveler still gets a usable plan even when one of the agents' tools fails along the way.

## The happy path

A run moves through the graph along a fixed spine with two conditional detours and one cycle.

First it validates the request — every destination city has to plausibly belong to the destination country. Then the Eligibility Agent works out the visa situation. If that answer is time-sensitive and the traveler gave a start date, a Deadlines step builds a real timeline (apply for the visa by this date, attend an interview by that date, and so on). Then the Logistics Agent and the Experience Agent run in parallel — one handles flights and hotels, the other handles restaurants, places to visit and activities. Both feed into a synthesis step that writes the briefing narrative. The graph then pauses at a review gate and shows the traveler the whole plan.

From the review gate the traveler can approve, revise, or cancel. Approving writes the visa deadlines to their Google Calendar and moves on to a second, separate gate that asks whether to email the finished itinerary. Only after that second approval does the email actually send. Cancel ends the run immediately, with nothing written and nothing sent.

## The multi-agent design

There are three agents plus the orchestrator.

The Eligibility Agent answers "do I need a visa, and what kind." It's a retrieval-augmented pipeline over a curated corpus of official government pages, with a live-search cross-check on top (described in the RAG section below).

The Logistics Agent handles getting there and staying there. It runs a real flight search and has an LLM recommend the best option against the traveler's budget and cabin preference, with pre-filled deep links to the usual aggregators as a fallback. It searches hotels for every city in the traveler's list — not just the primary one — and has an LLM curate and rank them per city against budget and preferences.

The Experience Agent handles what to do once you're there. For each city it pulls restaurants, places to visit and activities, and an LLM picks and justifies the best options for this specific traveler's stated preferences — dietary, family-friendly, outdoor seating — grounded only in the fields the data actually contains, so it never invents a dish or a view.

The orchestrator is the graph itself. It owns the routing, the parallel fan-out and fan-in, the retry and failure handling, the synthesis step, both approval gates, the revision loop, and the up-front validation gate. The agents don't know about each other; they read from and write to a shared state object, and the orchestrator decides what runs next.

A few control-flow details worth calling out. The Deadlines step is conditional — it only runs when the visa answer is actually time-sensitive and a start date exists. Logistics and Experience genuinely run in parallel and the synthesis step waits for both. The revise path is a real cycle: an LLM interprets the traveler's free-text feedback into a structured change (new dates, a new budget, or an add/remove/reorder on the city list), the affected nodes re-run — the deadline timeline too if the dates moved — and the graph comes back to the review gate showing exactly what changed. That loop is capped so it always terminates.

## Tool calls in detail

The system makes eight kinds of external call. Here is each one — what it does, what it returns, and what happens when it goes wrong.

**Pinecone (vector retrieval).** Used by the Eligibility Agent. The agent builds a query string from the traveler's nationality, destination, purpose and duration plus the specific question, embeds it with OpenAI's text-embedding-3-small model at 1024 dimensions, and queries a Pinecone index with a metadata filter that scopes results to the destination country and to documents that apply to this nationality or to everyone. Retrieval is hybrid — dense vector similarity plus a BM25 sparse score — fused and then reranked, pulling ten candidates and keeping the top four. Each result carries its text, source name and URL, a confidence tag, a topic tag, and the rerank score. If Pinecone or the embedding call fails, the exception propagates up to the Eligibility node, which is wrapped in retry-once and then records the agent as failed and continues.

**OpenAI embeddings.** A thin call to text-embedding-3-small, used at query time to embed the retrieval query (and separately at corpus-ingest time). A failure here surfaces the same way as a Pinecone failure — through the Eligibility node's retry-and-continue wrapper.

**You.com search API.** Used by the Eligibility Agent's freshness cross-check. It POSTs a query to the You.com search endpoint with the API key in a header, a twenty-second timeout, and gets back up to five web results, each with a URL, title and a few snippets. A missing key raises a distinct "not configured" error; a network or HTTP error raises normally. Either way the caller catches it, marks the cross-check as unavailable, and keeps the corpus-only visa answer.

**Duffel API (flights), over MCP.** This is the one tool exposed through the Model Context Protocol. A small server process publishes a single `search_flights` tool over stdio; the Logistics Agent spins up an MCP client, discovers that tool, and invokes it with the origin and destination airport codes, the dates, and the cabin class. The server wraps a real Duffel call — a POST to Duffel's offer-requests endpoint in test mode, with an outbound and a return slice, one adult passenger, a thirty-second timeout — and returns, per offer, the total price and currency, the operating airline and its IATA code, the number of stops, the duration, and the departure and arrival times, cheapest first. A few things about the MCP path needed handling. The result comes back as a list of text blocks, one per offer rather than one block holding the whole array, so each is parsed individually. A tool-level error inside the MCP server comes back as a text block that starts with "Error executing tool" rather than as a raised exception, so that string is detected and turned into a real exception. And the whole async call is bounded by a twenty-five-second timeout, because a spawned subprocess can hang indefinitely rather than fail fast — which was observed live, and which would otherwise defeat the retry-then-continue policy, since retrying twice is pointless if each attempt can block forever. If flight search is unconfigured or fails, the agent falls back to pre-filled deep links.

**Google Places API.** Used by the Logistics Agent for hotels and by the Experience Agent for restaurants, places to visit and activities. Each is a text-search POST that returns names, ratings, review counts, price levels, addresses, map URLs, and flags like family-friendly, outdoor seating and vegetarian options. If a search returns nothing, that city's section is simply empty and the run continues. If the call errors, the owning agent node retries once and then degrades — hotels or that city's experience block comes back empty, and the plan says so.

**Google Calendar API.** The first human-in-the-loop write. It needs an OAuth client credentials file in the project root; without it the tool returns a "not configured" status rather than an error, and the deadlines are still listed with a note that they would be written once access is set up. When it is configured, it loads or refreshes a cached OAuth token, creates one all-day event per deadline on the primary calendar, and skips any deadline whose date isn't a clean ISO date. It returns a status — written, no valid dates, no deadlines, or not configured — plus the list of what it created and what it skipped. The Deadlines node also makes an LLM call to extract the timeline from the retrieved visa sources in the first place.

**SMTP.** The second human-in-the-loop write. It uses Python's standard library — connect, STARTTLS, log in, send — with credentials from the environment and a thirty-second timeout. If the host, username and password aren't all set it returns "not configured"; if there's no recipient it returns "skipped"; a real failure like a rejected login or a refused connection is raised and handled by the sending node. The recipient is the address the traveler typed on the form, which is independent of the sending account — one configured mailbox can send to any address.

**Anthropic (Claude).** Every reasoning step. Nationality normalization, the visa answer generation, the corpus-versus-live reconciliation, the airport-code resolution, the flight recommendation, the per-city hotel and experience curation, the synthesis, the interpretation of revision feedback, and the up-front coherence check are all model calls. Every one of them that expects structured output goes through a single shared helper rather than calling the model directly. That helper tries the normal structured-output call twice; if the response is malformed it asks for raw JSON instead, extracts the object, repairs the two malformations that actually occur (a field whose value came back as a JSON string, and a field nested one level inside itself), and validates that; and if every route fails it returns a caller-supplied default, or re-raises if no default was given.

## Error handling in detail

Failure handling is layered, and each layer has a job.

**The graph never hard-stops on an agent failure.** Every agent node calls its agent through a retry-once wrapper. If the agent still throws, the node catches it, returns nothing for that agent's slice of the shared state, writes a "failed" entry into the run's trace, and appends a plain-language message to an errors list. The graph then routes to the next step as normal. The router after the Eligibility Agent still works if the eligibility result is missing. The synthesis step still runs whether or not Logistics and Experience succeeded. So a dead flight API, a down Places API, or a Pinecone outage each cost you one section of the plan, not the whole run.

**Synthesis names what's missing, and is itself defended.** The synthesis step is handed the errors list and whatever partial results exist, and is told to say in plain language what wasn't available rather than silently omit it. Its own model call is wrapped the same way as everything else: if it fails, it returns a short minimal briefing instead of crashing, and one of its output fields was made optional after the model occasionally omitted it on a partial run and tripped a validation error that took down the graph.

**Structured-output responses degrade instead of crashing.** The model sometimes double-encodes — it returns the whole object as a JSON string stuffed into one field — and a naive parse rejects that. Before the shared helper existed, a single garbled hotel-ranking response crashed the entire Logistics Agent, flight search included, even though the flight data was fine. Now that failure path retries, repairs, and finally falls back to a default chosen for graceful degradation: a bad hotel curation returns nothing and the caller uses the rating-sorted hotels without the written justifications; a bad flight recommendation returns nothing and the offers still render without a recommendation box; a bad airport resolution returns a crude three-letter stub so the deep links still build; an uninterpretable revision returns "couldn't read that, re-ran unchanged"; a failed coherence check returns "looks fine" so the validator fails open; a failed synthesis returns the minimal briefing.

**Configuration gaps are reported differently from real errors.** A missing API key is not treated as a failure. Flight search without a Duffel key says "real-time flight search isn't connected — use the search links below," and the pre-filled Google Flights, Kayak and Skyscanner links (plus a per-airline Kayak link for each carrier) are always built and shown regardless. Calendar without credentials lists the deadlines with "would be written once set up." Email without SMTP settings says "add the settings to enable it." The live visa cross-check without a key falls back to the corpus-only answer. A genuine error, by contrast, is surfaced as "this hit an issue" with the same fallback path — never as a stack trace, never as a silent no-op.

**Both write actions degrade like agents.** The calendar write and the email send are each wrapped in retry-once and a try/except that turns a failure into a recorded "failed" status plus an errors entry, and then the graph finishes. An approved write that fails does not crash the run; it just reports that it didn't go through.

**The validation gate fails open.** The first node checks that every destination city plausibly belongs to the destination country. If that check itself errors, planning proceeds. Only a confident "these don't match" stops the run — and then it stops cleanly, routing straight to the end with a plain fix-it message ("Toronto is in Canada, not Japan — change the country, or pick a city in Japan") and no agent having run.

**Revision is bounded and re-validated.** Feedback the model can't parse into a concrete change re-runs the plan unchanged and says so. A destination change is re-checked against the country before it's applied, so a revision can't recreate the Frankenstein-plan problem. And the revise loop is capped at two rounds — after that the model interpretation is skipped, the graph proceeds, and the UI removes the Revise control — so the cycle always terminates.

**The UI layer is defensive too.** Both the initial run and every resume are wrapped so an unrecoverable orchestrator error shows as a visible message rather than a stack trace or a button that silently does nothing. Text headed for Streamlit's markdown renderer is escaped for dollar signs first, because Streamlit treats a pair of them as math mode and had been mangling flight prices.

The net effect is that a run reaches the review gate with a usable plan under a wide range of partial failures — a dead flight API, a flaky model response, an unconfigured calendar, a down search service, an incoherent request — and the plan honestly says what's missing.

## RAG — the Eligibility Agent

The visa answer is retrieval-grounded. The corpus is a set of curated Markdown files of visa and entry rules, sourced from official government pages and indexed in Pinecone. Retrieval is hybrid: dense embeddings for semantic match plus sparse BM25 for exact-term match, fused and then reranked.

There are a few deliberate guards around the generation step. Before generating anything, a deterministic gate checks the top rerank score; if it's below a threshold, the agent refuses outright — "no source is a strong enough match to answer confidently" — without calling the model at all. A second check compares each retrieved document's topic against the traveler's actual purpose, so a student-visa page retrieved for a tourism question gets filtered out even though it shares the country and nationality. The traveler's nationality is normalized first, because the corpus is tagged by country name rather than demonym. When generation does run, it's told to use only the retrieved sources and to cite which ones it relied on. And immediately after generation, a grounding gate re-reads the answer against the chunks and strips or softens any claim they don't actually support — visa-waiver-program membership, fee amounts, law or proclamation names, where to apply — so the answer can't quietly import general knowledge that isn't in the sources.

The corpus has no automatic refresh, so on top of all that the corpus answer is reconciled against live You.com results. The rule is: if a clearly official live source — a government domain, an immigration authority, an embassy — addresses the traveler's exact nationality, destination and purpose, it's treated as more current and it drives the answer. Otherwise the corpus answer stands and live results only fill gaps. Every run returns a list of sources tagged corpus or live, and when the two disagreed, a plain sentence saying which one the answer follows and why.

## Evals

There are two eval suites, both making real calls rather than mocking the model.

### Agentic behavior — 18 / 18

`eval/run_eval.py` tests the properties specific to being an agentic system. A core section (11 checks) covers control flow, shared state, tool-failure recovery, both human-in-the-loop gates, the revise cycle, and that the replan loop terminates. A robustness section (7 checks) throws breaking inputs at it — typos in every field, a made-up nationality, a city that isn't in the destination country, gibberish — where the bar is graceful behavior, not an accurate answer, plus the corpus-versus-live reconciliation and the source tagging. Every check is a full run of the graph against live APIs.

All 18 pass. Highlights: the graph continues and still produces a briefing when the Logistics Agent fails twice; a deliberately malformed structured-output response degrades to a default instead of crashing; the calendar and email gates each block independently; Cancel ends the run cleanly; a date revise recomputes the deadline timeline and a destination swap re-points flights and hotels while keeping the other cities; the validation gate stops "Toronto in Japan" before any agent runs.

One caveat worth stating plainly: getting all 18 to pass in a single uninterrupted run turned out to be unreliable in this environment. The MCP flight subprocess occasionally hangs, and sustained API rate-limiting stretches a run from about twenty minutes to well over an hour or stalls it entirely. So the harness now prints and persists each check's result as it completes, and the 18/18 is assembled from that per-check record across runs rather than one clean sweep.

### Eligibility accuracy — golden set of 25 visa questions

`eval/run_accuracy_eval.py` is the Week 2 RAG harness pointed at this system's Eligibility Agent, against 25 labeled questions. It scores four things separately: did it reach the right conclusion, did it retrieve the document that controls the answer, is the corpus answer faithful to what was actually retrieved, and did the live cross-check leave a correct corpus answer alone, fix a stale one, or break a good one.

From the last completed run:

- Retrieval hit-rate: 100% (23 of 23 rows with a known controlling document). Hybrid dense-plus-BM25 retrieval pulled the right source every time.
- Answer-type accuracy: 71% (17 of 24 representable rows). This understates it. Several "misses" are the agent being more precise than the label: for Chinese-to-Australia, Chinese-to-USA-business and Canadian-to-Schengen-100-days the golden set says "visa required" and the agent says "a different visa category is required" — correctly, because those travelers can't use the ETA or the visa waiver and need a specific subclass. The Week 2 label schema is coarser than the answer.
- Live-reconciliation effect: mostly unchanged, with one row fixed and one broken. For India-to-UK the cross-check fixed a stale corpus refusal into the correct "visa required." For India-to-Japan it broke a correct "visa required" into a wrong "visa waiver with ETA," over-weighting Japan's eVISA pages. One flip each way — the exact trade-off of the "prefer an official live source" rule.
- Faithfulness was the weak spot. The judge's notes were consistent: the generated answers were stating precise facts the retrieved chunks don't contain — visa-waiver-program membership, specific fee amounts, the name and date of a 2026 entry proclamation, where to apply, validity windows. The model was filling gaps from general knowledge despite being told to use only its sources.

### The faithfulness fix

Three changes went into the Eligibility Agent in response:

- An in-pipeline grounding gate. Right after generation, a dedicated pass re-reads the answer against the retrieved chunks and removes or softens to "not specified in the available sources" any claim a source doesn't actually support — waiver-program membership, fees, law and proclamation names and dates, where to apply, processing times, validity windows — while keeping the conclusion and everything that is supported. This is the eval's own faithfulness check run as a guardrail inside the pipeline, not just as a metric.
- A tighter generation prompt, with an explicit list of fact types not to state unless a source contains them verbatim, and an instruction to keep the answer short.
- The same grounding rule applied to the reconciliation step, so the final answer can only state facts from the corpus answer or the live snippets it actually cites.

The accuracy harness was also corrected to judge the corpus answer against its own corpus-derived requirements rather than the reconciled ones. A clean re-measurement with these changes is still pending — the eval suite stalls under the API rate-limiting described above — but the fixes are in the code.

## Advantages

It's grounded rather than hallucinated. The visa answer is retrieval-backed, with a hard evidence gate before generation, a grounding gate after it that strips any claim the sources don't support, and a live cross-check on top. The flights, hotels and restaurants are rendered from the tool output directly — real prices, ratings and booking links — rather than being paraphrased by the model into prose, which in testing dropped and mangled links.

It's resilient by construction. A single failed tool or a single bad model response degrades to something usable instead of failing the whole run, and the output says what's missing.

The human-in-the-loop boundary is deliberate. Reads are autonomous; writes are gated. There are two independent gates, a revise path that's a real cycle rather than a restart, and a hard cap so it always finishes.

It's transparent. The sources behind the visa answer are shown and tagged; when corpus and live disagreed, the app says which won. Every revision shows the exact before-and-after of what changed.

And it reuses infrastructure honestly — the retrieval, corpus and index come from a prior project, and the new contribution is the orchestration layer around them.

## Challenges faced

The hardest lesson was that a structured-output schema is only as expressive as the fields you give the model. "Make Kyoto the main destination but keep Tokyo as a stop" is a reorder, and the first schema only had add, remove and replace — so the model did the nearest thing it could express and dropped Tokyo. The fix was to add an explicit "which city is primary" field and spell out the difference between a swap and a full replacement. The model understood the English fine; the schema couldn't represent the intent.

The structured-output calls themselves occasionally double-encode — the model returns the whole object as a JSON string inside one field — and a naive parse rejects that. Originally that crashed the entire Logistics Agent, flight search and all, even though nothing was actually wrong with the flight data. That's what drove the shared helper that retries, repairs and falls back, and a narrower failure boundary so one sub-step can't take down the rest.

The MCP flight subprocess could hang instead of failing fast — a run that just never returned, unlike the clean quick errors everything else produced. It's now bounded by an explicit timeout so the retry-then-continue policy still means something.

The synthesis step was itself a hard-stop for a while. Its output schema had a required recommendation field, and on a tool failure the model sometimes left it out, which failed validation and crashed the graph — the one node whose job is to gracefully report what failed was itself fragile. Made the field optional and wrapped synthesis in the same degrade-don't-crash pattern as everything else.

Incoherent input produced Frankenstein plans — destination country Japan with city Toronto gave a Japan visa check alongside flights to Toronto. That's why there's now a validation gate as the very first node.

Faithfulness was the weak spot in the accuracy eval. The cause was the generation step importing real-world knowledge it wasn't given: visa-waiver-program membership, fee amounts, a 2026 entry proclamation, where to apply. The fix was to add an in-pipeline grounding gate that strips any claim the retrieved sources don't support, tighten the generation prompt with an explicit do-not-state list, and apply the same rule to the reconciliation step. A clean re-measurement is still pending — see the note on rate limits below.

Once real email credentials were configured, the eval suite started actually sending mail to its fake test address on every run, which bounced into a real inbox. The eval now stubs the email sender for the whole suite.

Eval runtime was a real friction point, and by the end it became a blocker. Every check is a live call, and under sustained API rate-limiting a run that should take twenty minutes stretched past an hour or stalled outright — one accuracy re-run sat on a single row for forty-three minutes before I killed it. The MCP flight subprocess adding its own occasional hang made it worse. I added per-check pass/fail logging and an incrementally written report so a slow or interrupted run still produces data, and the agentic 18/18 is assembled from that per-check record rather than one clean sweep. A pending item is to make the eval resumable (skip rows already in the results file) so it can be finished in pieces.

A few smaller ones. The synthesis model would paraphrase or drop URLs when asked to put them in prose, so all links, prices and ratings moved to structured cards rendered straight from state, and the narrative is confined to the visa summary, the deadline timeline and a clearly-labeled recommendation. Streamlit re-runs the whole script on every interaction, so the graph and its checkpointer are built once and cached, and each browser session gets its own run id so concurrent users don't collide. And a free-text duration field could silently contradict the start and end dates, so duration is now just derived from the dates.
