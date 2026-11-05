# Explainer: teaching the tiny action model to keep its place

About a 3-minute read. Read [experiment 01's explainer](../01-pocket-actions/EXPLAINER.md) first for how the model works. The results section gets filled in after the GPU run.

## 1. The problem from 01

The 01 model completed 87% of tasks in apps it had never seen, but only 5% of tasks with 4 to 6 steps chained together ("delete X, then turn off Y, then move Z to W, then share V with Ana").

## 2. The likely cause, with an analogy

Picture the employee with sticky notes from 01. Each note had a number on it: note 1, note 2, note 3. In training, nobody ever handed them more than 3 notes. Then they get a stack of 6, and notes 4, 5 and 6 carry numbers they've never seen. They don't know what to do with them.

On top of that, the note-splitter had a bug: "and then" made it skip a number, so a 5-step job could produce notes numbered up to 8.

## 3. The fix

Instead of numbering the notes, we mark them: **done**, **doing now**, **next**, **later**. A 6-step job then looks exactly like a 3-step job: one note being worked on, one coming next, the rest later. The model only ever needs to learn "do the note marked *doing now*".

A note is marked done when the agent makes its final move for it: paste, delete, send or toggle. The agent keeps this memory itself, from its own actions. Nothing is read from the simulator.

We also save the models at half precision (float16). In 01, the full-precision files added up to 70 MB, over the 50 MB limit, even though an 8-bit copy scored the same.

## 4. How we'll know it worked

The pass bars are the same as in 01, fixed before training. The key one is **C4: at least 60% on long tasks**, up from 5%. C3 (at least 70% on unseen apps) must still hold.

As a check that the new signal is really the reason, we also run the trained model with every note marked *doing now*. If long tasks go back to failing, the signal is doing the work.

## 5. What we found

Pending the GPU run.

## 6. Further reading

- Andreas et al., [Modular Multitask RL with Policy Sketches](https://arxiv.org/abs/1611.01796) (2017): agents that follow a list of sub-tasks, keeping track of which one is current.
- Chevalier-Boisvert et al., [BabyAI](https://arxiv.org/abs/1810.08272) (2019): instruction-following grid worlds where "then" sequences are the hard part.
- Ross et al., [DAgger](https://arxiv.org/abs/1011.0686) (2011): the training method we keep from 01.
