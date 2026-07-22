## 30-Day Series

The `30-day-series/` directory contains a 30-day content series on LLM inference infrastructure. Every day file lives inside this directory and follows the same template and writing style.

### Template

Every day file must follow this structure exactly:

```text
Day N/30 of inference infrastructure

<title>

<small description about the previous conversation>

<deep dive into the current topics>
```

The template file is at `30-day-series/template.txt`. Do not deviate from it.

### Writing Style

The author's voice is conversational, direct, and momentum-driven. Match these traits in every day file:

* **Lowercase-first, informal punctuation.** Sentences do not need to start with a capital letter. Punctuation is minimal — no semicolons, no em-dashes for dramatic effect. Write the way you would talk to a friend who asked you to explain something.
* **Direct address.** Always talk to the reader as "you". Never use "one" or "the reader" or "users".
* **Short paragraphs, fast pacing.** Move topic to topic quickly. No filler. No padding. If a paragraph is longer than four lines, break it up.
* **Explain by example first, then name the concept.** Show what happens ("you ask chatgpt and you get the response back"), then explain why ("but what happens in between — thats what we see"). Do not define a term and then show an example. Reverse it.
* **No jargon gating.** Assume curiosity, not credentials. If a concept needs prior knowledge, say what to look up, do not say "you should already know this".
* **Honest scope-setting, framed positively.** State what the series covers. When narrowing scope, frame it as focus ("we focus on high level infrastructure") not exclusion ("this is not about kernels"). Lead with what you will do, not what you will not do.
* **Stream-of-consciousness flow.** The intro section of each day should read like spoken thought — natural, not rehearsed. The deep-dive section can be more structured with headers, code blocks, tables, and diagrams.

### Tone Rules

These are hard constraints. Violating them means the draft needs a rewrite.

1. **No negativity in the opening.** The first 10–15 lines of every day must feel welcoming and energizing. No words like "overwhelming", "hard", "struggle", "never give up", "disclaimer". If you need to set expectations, do it positively ("this series is focused on X" not "this series does not cover Y").
2. **No rhetorical put-downs.** Do not ask a question and immediately dismiss it ("does this require any knowledge, well no"). Either ask the question and answer it generously, or skip the question entirely.
3. **No apology framing.** Do not open with disclaimers about what the series is not. If scope is limited, state the scope positively and move on.
4. **Curiosity over credentials.** The only prerequisite is wanting to learn. Do not list things the reader "must know" in a way that makes them feel gated. If background is helpful, frame it as "if you have seen X before, great — if not, here is what it means".
5. **Forward momentum.** Every section should make the reader want to read the next one. End sections with a hook or a question, not a summary.

### Content Rules

* Each day must reference what was covered in the previous day in the `<small description about the previous conversation>` section. Day 0 is the exception.
* Code blocks, ASCII diagrams, and tables are encouraged in the deep-dive section.
* Keep the total length between 300 and 600 lines. Under 300 feels thin. Over 600 loses attention.
* Do not repeat content that was already covered in a previous day. Reference it with "we saw this on Day N" and move on.

### Agent Behavior for 30-Day Series

When asked to write or edit a day file:

1. Read the template at `30-day-series/template.txt`.
2. Read the previous day file to write the recap section.
3. Follow the writing style and tone rules above exactly.
4. If the draft has any negativity in the opening, rewrite the opening before presenting.
5. If the draft uses formal/academic tone, rewrite in conversational voice before presenting.
6. Do not add content outside the template structure without explicit permission.
