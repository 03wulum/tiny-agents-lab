# Explainer: a tiny model that operates apps without language

About a 5-minute read. The results section gets filled in after the GPU run.

## 1. What we tested and why

Can a model of a few megabytes, with no general knowledge and no ability to chat, complete multi-step jobs in phone apps it has never seen, like "move the newest PDF in Downloads to Taxes, then share it with Ana"? The smallest real-world model for this today is Google's FunctionGemma, at 270M parameters (about 0.5 GB). We test models 100 to 10,000 times smaller.

## 2. How it works, with an analogy

Think of a new employee who can't read full sentences but knows how to use a filing cabinet. You hand them sticky notes with the key words ("newest", "PDF", "Downloads", "Taxes", "Ana"). They look at what's on the desk, find the folder whose label matches a note, open it, and so on. They cross off each note once it's done. They never write anything back. They just point and act.

That's the **cue-and-pointer policy**:

- **Cues:** the instruction is cut into words. Each word is tagged with which part of the instruction it belongs to (its *clause*), whether it names a place to look (*location*) or a place to send something (*destination*), and whether it has already been used (a *tick*). Word order beyond that is thrown away.
- **Screen:** every on-screen item (app, folder, file, switch, contact) becomes a short list of numbers: its type, its file type, whether it's the newest or largest of its kind, and, most importantly, which cue words its label matches.
- **Pointer:** a small transformer (the same building block as ChatGPT, but tiny) mixes all this and picks one item plus one action: open, back, home, cut, paste, delete, share, send or toggle.

**Why matching labels matters:** in an app the model has never seen, every name is new ("hixyeltav", "oskjur"). A text model has to have learned those words. The pointer model only needs to notice that the label on screen equals a word in the instruction.

## 3. The process, step by step (and choices you might question)

1. **A fake phone.** PocketOS generates apps, folders, files and settings from a random seed, so every task happens on a phone the model has never seen. Six of the 20 app layouts, and a whole separate pool of names, are kept out of training entirely.
2. **A perfect teacher.** For every task, a search program computes the shortest list of steps. The model learns by copying it (*behavioural cloning*).
3. **Learning from its own mistakes (DAgger).** After the first round, the model plays on its own. Wherever it ends up, even somewhere the teacher would never go, the teacher says what the right next step is from there. This is repeated three times. Copying alone tends to fail, because one early mistake leads to situations the model never practised.
4. **Five sizes,** from 20 thousand to 5 million parameters, plus a "recursive" 2M model that runs its layers 4 times. All training runs only on Fortran's NVIDIA GPU, in 15 minutes or less in total.
5. **A fair rival.** A same-size model reads the whole screen and instruction as plain text and must *write* its answer as text, e.g. `CUT papalo.doc`. This is how language models do function calling today.
6. **Tests:** fresh phones (in-distribution), never-seen apps, tasks with twice as many steps, and instructions with new wording. A model only gets credit if it finishes within twice the teacher's step count.

Choices you might disagree with:

- The cue parser is hand-written and simple. It knows only "then" and "and" as joiners, so the new-wording test uses joiners it has never seen ("after that", "next").
- It's a simulator, not a real phone. That keeps it fast and fully measurable, but real apps have icons, scrolling and pop-ups.
- The text rival is small and trained from scratch, not a pretrained language model. The point is equal size, not equal knowledge.

## 4. What we found

_Filled in after the GPU run: the verdict, which criteria passed, the size curve, and what failed._

## 5. Why it matters

- **Phones and wearables:** a 2 to 8 MB action model fits in an app bundle and runs in a few milliseconds per step on a phone CPU, with no network and no data leaving the device.
- **Accessibility and automation:** "do this for me" across apps (files, settings, sharing) without a cloud assistant.
- **Agent design:** a small pointer model can do the clicking while a bigger model, if any, only plans. That cuts cost and latency.

## 6. Further reading

- Google, [FunctionGemma](https://blog.google/technology/developers/functiongemma/) (2025): the 270M on-device function-calling baseline.
- Chen et al., [Octopus v2](https://arxiv.org/abs/2404.01744) (2024): turns each function into a single token instead of spelled-out text.
- Erdogan et al., [TinyAgent](https://arxiv.org/abs/2409.00608) (2024): a 1B-class on-device assistant for Mac apps.
- Jolicoeur-Martineau, [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871) (2025): a 7M model that beats large models on puzzles.
- Ross et al., [DAgger](https://arxiv.org/abs/1011.0686) (2011): why learning from your own mistakes beats pure copying.

## 7. Questions someone might ask you

- **"Isn't this just a lookup table?"** No. Every test phone is new, and the unseen-apps test uses layouts and names the model never saw. It has to learn rules, such as "open the folder whose name is the location cue".
- **"Why not just use a small LLM?"** That's the text rival. At the same size, it has to spell out names it has never seen, which is where small models break.
- **"Would this work on a real phone?"** The decision part would. You'd need a screen reader (Android's accessibility tree already lists on-screen elements and labels) to produce the element list.
- **"How big is it really?"** The headline model has 2M parameters: 8 MB at 32-bit, about 2 MB at 8-bit.
