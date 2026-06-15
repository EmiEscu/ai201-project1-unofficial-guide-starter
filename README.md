# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

The domain that I choose was "Course and Professors Reviews at UIC". The reason why I chose this domain is because I am a rising transfer Junior at UIC and I am simply not familiar with the professors or the space at all. Sure I can go ahead a search each professor in Rate My Professor, but that will take too much time, and if I want to compare professors I have to make sure that they have similar classes. The reason why this knowledge is hard to find is simply because theres dozens of professors and remembering all of them and how they rank can be difficult. 
---

## Documents


| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | RMP | RMP Info for Abolfazl Asudeh  | documents\Abolfazl_Asudeh.txt |    
| 2 | RMP | RMP Info for Natalie Parde | documents\Natalie_Parde.txt |
| 3 | RMP | RMP Info for Jason Polakis | documents\Jason_Polakis.txt |
| 4 | RMP | RMP Info for Pat Troy | documents\Pat_Troy.txt |
| 5 | RMP | RMP Info for Luis Pina | documents\Luis_Pina.txt |
| 6 | RMP | RMP Info for Shanon Reckinger | documents\Shanon_Reckinger.txt |
| 7 | RMP | RMP Info for Sara Riazi | documents\Sara_Riazi.txt |
| 8 | RMP | RMP Info for Pedram Rooshenas | documents\Pedram_Rooshenas.txt |
| 9 | RMP | RMP Info for Anastasios Sidiropoulos | documents\Anastasios_Sidiropoulos.txt |
| 10 | RMP | RMP Info for Robert H. Sloan| documents\Robert_H_Sloan.txt|

---

## Chunking Strategy

     Since my documents are short reviews who are all seperated by a line spacing and are all equal length, I will split the documents into chunks by checking where there is a \n\n in the documents. In this case there will be no overlapping since the documents will all break off evenly. I will know if my chunks are too small or too large by simply looking at what is indside the chunk. All chuncks follow a format of Quality at the top, and word descriptions at the buttom, so all i need to look for is that those two thing are always at the beginnign and end of the chunk.

**Chunk size:**
Since these reviews are short and of equal spacing, to determine the chunk size I will simply look for where there is a \n\n in the documents.
**Overlap:**
There will be 0 overlapping since all reviews are seperated by a \n\n
****Why these choices fit your documents:**
Since I inputted the data manually, I know how the reviews are spaced out making it easier to include the whole review without being confined to character limitations or tokens.

**Final chunk count:**
224 chunks across 10 documents
---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model Used:**`all-MiniLM-L6-v2`, run locally through the `sentence-transformers` library (configured in `embed.py`). It produces 384-dimensional embeddings, and I normalize the vectors (`normalize_embeddings=True`) so they can be compared with cosine distance, which is the metric I pinned on the ChromaDB collection (`"hnsw:space": "cosine"`). The same model embeds both the chunks at index time and the user's query at retrieval time — that match is essential, since mixing models would make the distances meaningless.

I chose it for three reasons that fit this project: (1) it runs **locally with no API key, no cost, and no rate limits**, which matters for a student project I'm rebuilding and re-querying repeatedly; (2) it's **small and fast**, so embedding ~all of my review chunks and answering a query both happen in well under a second on CPU; and (3) it's **accurate on short English text**, which is exactly what RateMyProfessor reviews are — a few sentences each. A heavier model would be overkill for chunks this short.

**Production tradeoff reflection:**

If I were deploying this for real users and cost weren't a constraint, I'd weigh moving to a stronger hosted embedding model (e.g. OpenAI `text-embedding-3-large`, Voyage, or Cohere) against the simplicity of the local setup. The tradeoffs:

- **Accuracy on domain-specific text.** `all-MiniLM-L6-v2` is a strong general-purpose model but a small one (384 dims). Larger models (1024–3072 dims) capture finer semantic distinctions — they'd be better at separating queries like "is the *grading* harsh" from "is the *workload* heavy," and at the professor-vs-concept disambiguation problem I hit in my failure case. That's the upgrade most likely to improve answer quality here.
- **Context length limits.** MiniLM truncates input at 256 word-pieces. My chunks are single short reviews, so this never bites — but if I later chunked long-form content (subreddit threads, syllabi), a model with an 8K-token window would avoid silently cutting off the tail of a chunk.
- **Multilingual support.** MiniLM is English-only. UIC reviews are essentially all English, so this doesn't matter today; if the corpus expanded to multilingual student posts, I'd switch to a multilingual model (e.g. `paraphrase-multilingual-MiniLM` or a hosted multilingual embedder) so non-English queries and documents land in the same space.
- **Latency.** Local MiniLM has no network round-trip — embedding is effectively instant. A hosted API adds 50–300 ms per call plus failure modes (timeouts, rate limits, key management). For an interactive UI, the local model is actually the *better* latency choice; the hosted model only wins on accuracy.
- **Local vs. API-hosted.** Local means zero cost, full privacy (reviews never leave the machine), and reproducible offline runs, at the price of weaker embeddings and using my own compute. API-hosted means better embeddings and no local model weights, at the price of per-query cost, a network dependency, and sending data to a third party.

**Net:** for a real deployment I'd most likely keep embeddings **local but upgrade the model** (e.g. `bge-large-en` / `gte-large`), which buys the accuracy of a bigger model while keeping the no-cost, low-latency, private profile — and only reach for a hosted API if I needed multilingual coverage or to offload compute entirely.

---

## Grounded Generation

**System prompt grounding instruction:**

Grounding is enforced in two layers — an *instruction* layer (the system prompt) and a *structural* layer (the code) — because a prompt alone can be ignored by the model.

The system prompt (in `app.py`) tells the model it is "The Unofficial Guide" and may answer using **only** the student reviews in the `CONTEXT` section. Its strict rules are:

1. Use ONLY the information in the provided CONTEXT — no outside or prior knowledge about the professors, courses, or UIC, even if the model "knows" something.
2. If the CONTEXT doesn't contain enough to answer, reply with exactly: `"I don't have enough information on that."`
3. Don't invent professors, classes, grades, or quotes — every claim must be supported by the CONTEXT.
4. Treat reviews as student opinions, not facts (phrase as "students say…", "several reviews mention…").
5. Cite sources inline with the bracketed numbers from the CONTEXT, e.g. "students found the workload heavy [1][3]".
6. Be concise (1–4 sentences).

The user message reinforces this: each query is wrapped as `CONTEXT (student reviews retrieved for this question): … QUESTION: … Answer using ONLY the CONTEXT above`, and generation runs at a low temperature (0.2) to discourage the model from "filling gaps" with its own prose.

**Structural choices that the model cannot override:**

- **Context is the only knowledge in the window.** `build_context()` formats the retrieved chunks into a numbered `[1]…[k]` block, and that is the *only* content placed in the prompt — there is no other professor knowledge for the model to draw on.
- **A relevance gate before generation.** Before the LLM is ever called, `answer_query()` checks the best retrieved chunk's cosine distance. If nothing is retrieved, or even the closest chunk is farther than `MAX_RELEVANT_DISTANCE` (0.95), the system returns the refusal directly and never calls the LLM. This stops confident answers to off-topic questions ("what's the weather?") at the retrieval stage.
- **Refusal short-circuit after generation.** If the model does answer but emits the insufficient-information sentence, the code returns the refusal alone — it won't attach sources to a non-answer.

**How source attribution is surfaced in the response:**

Attribution is surfaced as a **Sources** list appended below every grounded answer, and it is built **programmatically** rather than trusted to the model. After generation, `format_sources()` reads straight from `retrieve()`'s output and emits one entry per retrieved chunk, numbered `[1]…[k]` in the **same order** `build_context()` fed them to the LLM — so the inline `[n]` citations in the answer line up with a visible, inspectable source. Each Sources line shows the **professor**, the **source document** (e.g. `Natalie_Parde.txt`), a **relevance score** (`1 − cosine distance`, higher = closer), and a short snippet of the actual review the claim was drawn from. Because the list is generated from the retrieved metadata rather than the model's text, every answer is attributable to real documents by construction — the user can see exactly which of the top-k chunks the response drew on. When the system refuses, no Sources list is attached, since there is nothing real to attribute a non-answer to.
---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Is workload something I should worry about when walking into Polakis class? | Multiple reviews state how his class is Lecture heavy, has a lot of homework assignments, and the material is super heavy | Students say that the workload in Professor Polakis' class is heavy, with several reviews mentioning "Lots of homework" [1][3][5]. One student notes that the amount of homework and projects is "unbelievable" [5], while another states that the homework is "tedious and hard" [1]. Therefore, workload appears to be something to worry about when taking a class with Professor Polakis.| Relevant | Accurate |
| 2 | Is it easy to get a hold of Anastasios Sidiropoulos if I ever get lost on any assignments or have questions? | Most reviews talk about how caring this professor is and how he is willing to help anyone who has a question for him, but there are also some reviews that talk about the difficulty of the course and how his course is intimidating. | Students say that Professor Anastasios Sidiropoulos is "approachable for questions" [1], which suggests that it is easy to get a hold of him if you have questions or get lost on assignments. | Partially relevant | Partially accurate  |
| 3 | What is one praise most of Prof Pina students have about his class structure? | Most reviews talk about how they love how incerdibly organized class is. | Students praise Professor Pina for having a well-structured course, with several reviews mentioning that his lectures, labs, assignments, and exams "seamlessly tie into one another" [2], and that he is "organized" [1] and has "clear grading criteria" [2][4]. | Relevant | Accurate |
| 4 | Which professor has the highest overall ranking? | Based on the provided information Luis Pina has the best overall ranking of all professors. | Students consistently rank Professor Luis Pina highly, with all reviews giving him a Quality rating of 5.0 [1][3][4][5], indicating that he has the highest overall ranking among the professors mentioned. | Relevant | Accurate|
| 5 | Are there any exams in professors Natalie Parde class? | There are comnflicting reviews for this, some say that there are exams but dont really specify the amount. One thing that they do mention is how long the exams they have are and how they can be tough. The reviews also talk about homework assignments and semester long projects that are easy to follow thanks to the instructions and resources she provides. | Students mention that Professor Natalie Parde's class, CS421, is "Test heavy" [1], indicating that there are indeed exams in her class, along with a project and 5 assignments. | Partially relevant | Partially Accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis


**Question that failed:**
Are there any negative points reviews bring up about parde teaching style and her class structure?

**What the system returned:**
The system returned a pretty positive response: Students do not mention any negative points about Professor Parde's teaching style [1][3], instead, they praise her for being clear with her expectations, giving good feedback, and being accessible outside of class. The course structure is also described as straightforward with clear grading criteria [3]. No negative comments are made about her teaching style or class structure in the provided reviews.

At first glance there seems to be nothing wrong with this, there are sources for where it got this and it even sounds convincing. The real issue falls when you look at the sources themselves. Most sources that it retrieves are not from the Natalie_Parde.txt file and are simply from other professors documents, but it does accurately index only the ones for Natalie Parde. The issue is that the reviews being quoted have nothing negative about Natalie. 
**Root cause (tied to a specific pipeline stage):**
The root cause is the embedding and retrieval. other professors dominate the top-k. This can be because the metadata requires professors name to tie Parde to the correct chunks to retrieve and the system doesnt specify to only retrieve chunks for such professor.


**What you would change to fix it:**

By simply changing the name from just including 'parde' to including 'Natalie Parde' the system retrieves a review that matches these 'negative points' the user may be looking for with their initial input. Since the system retrieved better relavent chunks that were more relevent to the question with the full name of the professor, the retrieval code will likely work better if it connects lastnames or shorter names for a professor to connect vibes with a specific review. 

Resolve the professor named in the query (map parde → Natalie Parde), then pass where={"professor": "Natalie Parde"} to collection.query(...). Now top-k can only come from her reviews — deterministically, regardless of wording.
Optionally also prepend the professor name to each chunk's text before embedding, so identity is part of every vector instead of inconsistently present. This helps ranking but, on its own, is still softer than a hard metadata filter; the filter is the clean solution
---

## Spec Reflection


**One way the spec helped you during implementation:**
The Specs helped a ton. Simply having them there made it easier to understand the end goal of the program. It was like coding but in advance pseudocode. Having the specs done made it even easier to code the program since Claude also had a much more clear image of what it was supposed to build and a pipeline/architecture of the system.
**One way your implementation diverged from the spec, and why:**
One way my implementation diverged from my specs was by adding a lot more ways to verify that the outputs such as chunks and retrieval chunks to a query were accurate. I made sure to add a bunch of functions that would allow me to test these milestones to make sure the results were what I was looking for before moving on.
---

## AI Usage

**Instance 1**

- *What I gave the AI:*
     I gave Claude the results app.py returned the first time I ran it as I was confused on the reasons why there were indexes in the output and what they meant.
- *What it produced:*
     It returned a summary explaining that these indexes really had no use since the retrieval chunks they were citing did not appear within the UI, so the user simply saw random indexes.
- *What I changed or overrode:*
     Since it had already added the indexes, I thought it would be a good idea to continue within that route and I made sure to tell it to change the code to add a way to see those retrived chunks that were cited so that the user has a better understanding of where they come from.

**Instance 2**

- *What I gave the AI:*
     I gave Claude my chunking strategy that was withing planning.md as well as the pipeline I was going for the project.
- *What it produced:*
     It produced a working chunking python script as well as a chunks.json file that was used to store all the chunks that it generated. 
- *What I changed or overrode:*
     Since it had all the chunks generated in a .json file it was difficult to sort through them and see what was what, so i continued to prompt it to add more features to allow for more testing. For example, seeing all the chunks for one professor or simply looking for a specific prompt for that professor. 
