# otakuscan

> **Status: On Hiatus**
>
> Well, this took a turn.
>
> otakuscan was my attempt of BYTE's Applied ML - Advanced Task,
> 
> _i request the awesome person reading this, to read till the end :)_

## What is this?

otakuscan started as my attempt to make a machine read manga properly, read the panels, findout who's speaking, and maintain character identity and thoughts with the narrator.  
Simple on Paper, right? i guess not, since its on Hiatus now


## What I've built so far

- **OCR pipeline:** Uses RapidOCR to extract text from manga pages.
- **Reading-order detection:** Uses spatial and panel-aware heuristics to arrange text in the correct order.
- **Learned components:** Includes classifiers for filtering irrelevant text and identifying narration.
- **Speaker assignment:** Uses conversational and spatial heuristics to maintain speaker consistency.
- **Evaluation pipeline:** Includes prediction scripts, validation tools, and integration with the task's official scorer.

The project ( and I ) went through several iterations, with different prompts to text extraction, filtering and speaker classification.  Somehow, every iteration worked better than last.

## Why the hiatus?

Honestly, the project started moving faster than I could keep up with. 

I got a collection of prompts* working together and learned hell lot along the way. But i guess there's a difference between prompting something to just run and finding out why it works, breaks, & how to make it better. I need to learn more i guess :/

So, rather than keep piling on more prompt-n-code and hoping that one thing clicks, I'm taking a step back to actually learn things.

## Where things stand

The last version uses an OCR-based pipeline with  text-filtering part and speaker assignment. It was jusst an experiment, not something world class*

For now, development is paused while I work on the other two tasks lol

If I return to otakuscan, the plan is to revisit the existing model, understand its weak spots, and make improvements rather than blindly prompting.

Maybe I'll come back with a clearer head and a better plan. or it'll return just like [this](https://www.imdb.com/news/ni64994621/) years later

For now, that's otakuscan. Not quite the ending I had in mind, but lets not spoil the mood :)

---

**P.S.** I was just vibe coding lmao. Turns out, prompting your way to something at [Advanced Level](https://bytesoc.dev/tasks/applied-ml/#:~:text=The%20Manga%20Task%20%5BAdvanced%5D) and actually understanding how the thing works are two very different boss fights.

Signing off~~ 

Lakshay Malik ㅤㅤ -aka-  _**neolakshay**_
