# Nordling Meeting @Today 7:45 PM

Summary

### Action Items

- [ ]  Esteban will send pseudocode update via Asana
- [ ]  Verify how Python implements certain functions (ask Claude with references)
- [ ]  Explain in the cover letter why the numbers changed compared to the previous version
- [ ]  Make code available as .py functions (not Jupyter notebooks) in the public repository
- [ ]  Add a note in the paper about memory limits when the number of combinations is large
- [ ]  Review the size of V for different resolution cases

### Notion AI Transcription Test

Notion's AI functionality was tested for taking notes and summarizing meetings. It was mentioned that tools like the one Jamie Neistat uses only work with educational or specific enterprise accounts, not personal accounts.

### Algorithm Comparison Results

**Experiments conducted**: Three experiments were run — real data (A), B1 (half-circle and full circle), and B2.

**Results by experiment**:

- **Real data (A)**: Civica wins, followed closely by Greco and Chi; RFCA in fifth place after 3CFBI
- **B1**: 3CFBI wins in these artificial experiments
- **B2**: 3CFBI also wins
- **Low resolution (RFCA)**: RFCA performs better when quality is very poor

There is no algorithm that is always superior — the winner depends on the zone and specific conditions.

### Tests with ChatGPT and Gemini

The pseudocode was given to ChatGPT and Gemini to recreate 3CFBI, but they are not achieving results as good as the current implementation. The Jaccard index for real data was much lower and very similar between both.

### Data Visualization Decisions

**Resolutions to plot**: It was decided to use 480, 240, 120, 60, 30, 15 (6 points) instead of 480, 240, 120, 160 to make the numbers more readable in the graphs. This allows increasing the size of the numbers in the visualizations.

**Table precision**: Do not increase tables to 4 decimal places — keep 3 decimals and mark the winner in bold.

**Resolution terminology**: Use "half" instead of "50%" to describe resolution reductions, as it is clearer. For example, 240×240 is half of 480×480, not 50% or 25%.

### Detailed Pseudocode Review

**Line-by-line — Clarity improvements**:

**Line 6 — Triplet sampling**:

- An array is created with all possible triplet combinations, then sampling is done
- Time complexity can be problematic when the number grows — with 200 edges there are 1.3M combinations, with 500 there are 20M and 3.6 GB of memory is needed
- Currently not a problem because there are typically ~150 pixels (more than 100,000 combinations)
- The number at which more than one gigabyte would be needed should be specified

**Line 11 — Rounding**:

- Removed the word "cell" and replaced it with "element"
- The rounding method must be specified: NumPy uses "banker's rounding" (round half to even) — 1.5→2, 2.5→2, 3.5→4, 4.5→4
- New wording: "Round x, y, r to nearest integer using banker's rounding and increment element v[x,y,r] by 1"
- Reason for banker's rounding: avoids bias when using discrete arrays

**Line 4 — Data structure**:

- V changed from being an array to a Counter (map/dictionary type)
- This makes V a sparse object — it only stores elements that appear, not zeros
- New initialization: "Initialize an empty vote map V(x,y,r, count)"

**Line 13 — Peak selection**:

- Changed from "cells" to "elements"
- "Let P be the n elements of V in decreasing order of count"
- Ties problem: ties will occur frequently and how they are handled must be specified
- The iteration order and how the top peaks are found needs to be specified

**Line 0 — Input parameters**:

- Change "sample size" to "triplet count" for clarity
- "Center tolerance" renamed appropriately
- Cube half-width (τ) changed to "cube side length" (κ=3) to simplify and avoid confusion about whether it is a radius or half-width

**Lines 16–18 — Weighted centroid**:

- Notation updated to be consistent with the map structure
- New notation: "Initialize vote map V(P,C) with P = (x,y,r) and count C"
- On line 11, change "count" to just "C" while keeping x,y,r explicit for clarity

**General formatting improvements**:

- Add whitespace between "for" and elements to improve readability
- Notation consistency: use square brackets for closed intervals, parentheses for triplets — do not mix with tensor notation
- Avoid introducing non-standard terms such as "map" used informally
- Remove the word "sparse" from initialization since it is not presented as a formal sparse object

### Reproducibility Concerns

The main problem is the inability to reproduce one's own results. This occurs because:

- The algorithm has not been correctly defined to match the code implementation
- Changes in the code produce changes in results without a clear record of what produces what

**Proposed solution**: Be extremely strict — define the algorithm exactly as it should be, ensure that the implemented code corresponds exactly, with no additional things.

### Mathematical Notation Consistency

- Do not use terms informally — use standard mathematical forms
- Do not use code notation (such as C) directly without explanation
- Minimize the risk of offending reviewers with details that obscure the work

### Personal Context

A letter was received from Ray after a long time without contact. Ray mentioned that the new PhD student is using Esteban's desk.

### Next Technical Steps

- Continue refining the pseudocode with the corrections discussed
- Verify specific Python function implementations (Counter, rounding) by consulting documentation
- Ensure the public repository has code as callable functions
- Complete the cover letter explaining differences in results versus the preprint version

---

Notes

Transcript

Subtitles made by the Amara.org community

Subtitles made by the Amara.org community

Until next time!

How are you today? Have you had a good day at work?

Uh... yes, so I'm working a lot from home — at this point I'm like a... uh...

2 or 3 months. I think I've made good progress. Yes, I'm happy. Just so you know, I'm using Notion to remember and take notes from this meeting. I want to try the AI. The one Jamie Neistat uses is only for Gmail related to education and some specific companies — if it's a business account, for example.

It's not for regular people's accounts, which is why GemNade doesn't work with that. Yes, but with Notion I understand that... it's doing the work right now, so I'll show you after. Ok, great. Please tell me how much you like mobile. Because there is a big difference in quality between the different ones. Yes, I assume so.

What else? So I saw you right in the middle. Yes, yes. So I worked... on the SOTU code, on the code itself and checked the English several times and everything. Everything should be referenced in the document without having repeated information, as it sometimes does. After working on the algorithm, on the pseudo-code, for some... like some loops, I gave it to...

ChatGPT and Gemini to see if they could recreate the 3CFBI, and they are not getting results as good as ours. It's my code, so I don't know right now when and why. And so we have three experiments: A — Real data, B1, and B2. B1 is half a circle and then the full circle in another paper, and B2 which is the hour.

And 3CFBI wins in these two artificial ones. Civica wins in the real data. And when we compare — yes, so some of it makes sense, but I was hoping to have one that is always better, but it seems that sometimes, depending on the zone, one is better. Some win and some don't.

But for which combination of resolution and noise level does Civica win and 3CFBI win?

Hmm, ok.

Did you see this?

I have some questions for you, so...

If we come here to look at experiment A, Real World, Real Data, Civica wins and Greco and Chi are very close. And RFCA is like in fifth place, after Civica, Greco, Chi, and 3CFBI. I mentioned RFCA because the other B1 results show 3CFBI winning — this is the number of ties.

Yes? Do we normally win? And if you had a 4-4 optimal, which one would win? What? If you had a fourth-tenth, which one would win? Chi or 3CFBI? Oh, I don't know.

...I can do that. I'm getting ready to take notes here.

So this is across different outlier counts, but...

So, do you think comparing with a fourteenth is a good idea?

Yes, let's say we win, I think it's... I don't think it makes sense to say we win.

When I make the table, I wouldn't increase it to 4 decimal places — I'd keep it at 3 decimals, but just put the winner in bold. Ah, ok!

A greeting.

Because, after all, this is an artistic example.

Let me show you something. Are you working today? Yes. I received this letter this week from Ray. Yes, yes. I was very surprised. What is she saying? That she misses you? Yes. It has been a long time since we talked. She sent me this.

March 21st and it arrived around May 8th. So yes. She told me the new PhD student is using my desk. I hope it brings him good luck.

Yes, before I forget that. I have a question here. Resolution infinity. Resolution 480×480. Somehow 480×480 is the good resolution.

And this 240×240 represents...

And downscaling 50 by 50 percent. Yes, this is not important, but would you say 50% or 25%? Because it's square.

This is a quarter of it in number of pixels.

I would just say the resolution is half. Ok, ok, so 50% is half. No, I don't use the term 50% — I would say it's half. It's unclear what the reference is with 50%, and there are more unclear things about how it's calculated, but if you say half... it is, in my opinion, clearer. Ok. For the next 2 minutes, I'll use the percentage only with you. You have to continue with this table.

I understand what you're saying, but we have the same language. When we go to 12×12, that is... 2.5% or 1 divided by 40, right?

So this is very small, very poor quality. And I'm asking you this because when we come over here...

I added one of these.

I think I added 160×160, this one.

Yes. And here, somehow... Can you see the numbers? No. Yes, if you do it like this, yes. So, how do I fit the numbers clearly here if they don't fit? Should I make the... Can I plot further than the text?

And in this case, RFCA is winning when it gets very bad. Yes, I think so. But wait, why do we now have 480, 240, 120? Why don't we have 60? 30, 15, why do we have these numbers? These are the numbers I've always used. But why don't we have the things from A?

Sorry, I'm coming back.

Hello? Yes?

Ah yes, I'm Esteban, but it's fine, I'm in the meeting, so come find me later.

Ok, sorry, I don't remember what the criteria was, but we had...

We remember these numbers.

If we have half, like 480, 240, 120, 60, 30, and 15 — that's 6 plots — and if I have 6 plots I can make this number much larger because I can... We need to have... yes? Exactly, that was my point. Because then we can make it readable. Yes.

I put it 3 times 2, I take it out and instead of this... yes yes I like that because the other is worse. So here we see that 137, 3CFBI winning, Civica, Chi is good, and RFCA — that was very bad. Not very bad, but it wasn't very good with the...

Real data comes in second place. And this plot here, oh, yes, so this will be much better. This is only 3CFBI. And... Yes, so...

We would have 15 and that would make it much better. Ok! I like that!

So, when we drop...

How do you know which method is doing what here? These are the only results for 3CFBI.

Ah, ok.

So that is the 120 version.

Subtitles made by the Amara.org community

Thank you for watching the video!

Thank you for watching the video and for sharing it with your friends! I wanted to discuss or show you, and just as you can see here, I'm going to...

Present something else... This is the...

Ok, V3 is the one we're using, just in case. Other steps? You know that in the cover letter we have to explain why the numbers are different in the previous image compared to this one.

I hope they know why they are different. Did the code change? Yes, but we can't write that the code changed — we have to write why and how the code changed. Okay, the preprint that is in the archive — I didn't find an explicit SOTU code available there.

So you need to analyze the real code and compare it to the new one.

In the repository that we make public, we have to make the code available as functions — not as Jupyter notebooks. Yes, yes! I have... The things I'm running are .py

that are callable — callable, I don't know how to say it, but if name equals name...

Name equals name? I mean, basically it's just function calls. So you call 3CFBI and then you input the data. Yes?

So here, as you can see, the Jaccard index for real world data, for ChatGPT and Gemini, is much lower. Very similar between them — both used the algorithm. And what coding error did they make then?

I had that before... I... wait.

Hello. Greetings. Ah, yes. Still in the meeting just in case, but ok.

Ok, and...

B2

Why this now?

I have to review this — it's not what I expected, but yes, I don't know why. I've been trying to figure it out. And because I was expecting the code, the algorithm to be clear. Ah, let me show...

Can you read that? Yes, I can read it.

Lovely!

Did the stove arrive? Did it? Yes.

A greeting.

First you generate every possible triplet and then you sample from that set?

How do you implement the method? I create each triplet of N without needing... repetition, and that is set as a language array of 5000, typically, times 3. And then for each of these columns indicating one

triplet of coordinates, because each number represents a point and a point has 3 coordinates.

Uhm, yep.

But generating that entire NumPy array with every possible triplet — that could be extremely large in some cases, in memory. No, no, I only generate 5,000... randomly.

No, no, I don't do it that way.

Yes, but then you're not doing this minimum, because for the second case — where you have the number of edges over 3 — that's the one you should do deterministically to avoid repetitions, because if you draw from that distribution you'll have repetitions.

Well, in the first case too, you could end up having repetitions if you draw uniformly. And you draw randomly, actually.

How is it implemented uniformly?

And also when you do the minimum — when the number of edges over 3 is less than 5,000 — then you actually have to generate all of them to avoid repetition.

So it would be good to have a whitespace between "for" and each element.

Also at the top, in line 6. So both suffer from the lack of whitespace.

Subtitles made by the Amara.org community

So, in fact, I create all possible combinations, and then from there... Uhh...

From there, I quickly take...

N! And N is the minimum of this.

Okay, which means that there will be horrible time complexity. That's fast!

It's fast, as long as the number is small, but when that number starts to go up a little, the thing will explode completely because it grows faster than exponentially. Yes, in my case, I think the largest...

I thought I had that. Ok, in my case I moved 150 pixels. So it's 150 over... a tree, and that was something around... more than 100,000.

Yes, so those are still very small numbers. But that, you know, will grow very, very quickly.

So that deserves — I mean, we want to publish and I don't think people will prevent us from publishing because of that detail, but we have to make a note in the paper about this, and then say that if the number is large, then this could be replaced by another sample.

With a sampling strategy to avoid taking every triplet.

If we have 200 edges, the number of combinations is 1,300,000.

If it's 500, it's 20,000,000 and then you start needing 3.6 GB of memory. And it scales up fast. Yes, indeed. It wasn't a problem because we were always... We were around. Yes, so you need to mention, for example, the number at which more than one gigabyte would be needed.

OK.

What is an integer cell?

9

Here. Line 11. Yes. So why do we get X, Y, R? No, no. An integer set.

I've never heard of an integer cell before. Ok, but do you understand what it means?

I can guess, but why introduce a new concept that has never been used when there is no need to introduce it? Because for me it's simply... So I don't feel I'm presenting a new definition.

Thank you very much.

The thing that is not clear is how you round the values, and then you introduce the integer cell, which is not necessary. If we remove the word "cell" from the phrase, it's more understandable, but there is still the problem that it doesn't define how rounding is done.

Do you take the ceiling? Do you... approximate, round is...

Then you have to say how you round.

So, round to the nearest integer. I think that is...

No, that leaves the question of how you round. If a value is, for example, 1.5, 0, 0, 0. How is it rounded then? Is it rounded up or down?

I don't know what Python does — I think if it starts with an extra number it should go up, even should go down in the case of a 5 afterwards, but I don't know. So I know there is a criterion, I don't know which one Python is using, but I really think...

I really think you need to specify it. I mean, it's if...

You have two cases. If it's greater than 0.5, it rounds up, for example, or greater than or equal to, it rounds up.

I mean, there's no point in having uncertainty on such a thing — just specify it.

So suddenly you use C code notation there with this V, X, Y, R. Yes. Yes. Yes. Yes. Why do you say yes?

Well, that's a notation that I think was originally introduced in C, or maybe it was already introduced in Fortran before that. I'm not sure, actually. I would say today that this is a coding thing that you can use in... in Python you can do this, so I assume that anyone who codes will understand this. I don't know why...

Uh...

The devil is in the details, and we just need one detail to obscure a reviewer. If we can avoid having any detail that risks offending someone, we should.

I completely appreciate that, but I really think...

I have to disagree with you. I don't think this level... I don't know. Well, what should I write here?

Well, I'm trying to think about this. Are we consistent? Can you scroll up a little, please, so I can see the definitions at the beginning? You don't need to go this far, just keep it so that it's... yeah, thank you, perfect.

Ok, so here we define a closed set for Rmin and Rmax, so we use square brackets to indicate a closed interval. An interval, yes. Yes, a closed interval. Instead of an open interval. A greeting.

Then you use square brackets to indicate...

an element in a tensor. There. Yes. That's what it is. It's an element in a tensor. Yes. Right.

Then we use parentheses to indicate the triplet. Where are you looking? Here? Yes? And nothing...

And the V that is introduced there in the setup button.

You said it was a map.

Something that transforms from one space to another. A function that transforms from one space to another. Why is that a map? That's not mathematics.

It's just an array.

Why do you use "map," Esteban?

Somehow, I still don't see it. You thought it sounded good. What? You thought it sounded good. Yes, and I don't know why. For me, "map" mapping — a map is not always good. A map of one place to another, so in English I assume a map is what the...

A map, you know? Google Maps — that's also a map and it's not pointing from one place to another. Yes, but we're making a map here, so we have to be consistent in the definitions. We have to choose a common notation and then follow it strictly, because otherwise we'll confuse people.

And we shouldn't use terms in a colloquial way — we should use them in the normal way they are used in mathematics.

So, Integer Vote Array — we can speak of an array, it's a well-defined object.

Should I remove the walls? Doesn't add much value?

Are you presenting it as a sparse array? You know there is a special data type for sparse arrays in certain languages. In MATLAB I use Sparse a lot. In this case, it's not a code thing to use Sparse for V — it's simply saying the idea is to understand that, at the end, many of the values remain zero or empty.

A tensor with zero values and at the end a bunch of zeros remain — in that sense it's sparse, but...

But then you don't present it as a sparse object, so it shouldn't be called sparse there, because here you say strictly to initialize an empty object. And if you say empty, then it must be an empty object.

So remove the word sparse. Done.

And... yes, so they tell me: Initialize an Integer Vote Array.

Regarding the rounding, NumPy rounds always to even numbers, so 1.5 goes to 2, 2.5 becomes 2, 3.5 becomes 4, 4.5 becomes 4 — and it seems it's called Banker's rounding, round half to even. So if you want... Yeah, but then you should mention that.

Banker's rounding then, if that's the term for it. I didn't know NumPy was implementing that — I find it quite peculiar. Why would they have selected such rounding? Because in mathematics that's how it's done — if you go to equal or odd, I don't remember, and it seems you always go to even.

But I would still say that the most common thing is that if it's 0.5 you always round up, and if it's below you round down. I hadn't thought Python would implement banker's rounding, so it's good to know — I've learned something new and I need to remember it. Distortion of the distributions.

But we haven't said it's an array.

If you added and selected element v[x,y] which is positive equal to 1. No, but then instead of writing this plus equals one, you can just write: increment element v. x, y, r by 1. Then we have it in language without depending on specific notation.

I didn't hear everything you said, but... increment the element. So if you use the verb "increment" in front of it, then it's clear.

... ... ... ...

But after this rounding, you also need to add banker's rounding. Yes, so that's what I'm trying to... because rounding appears here as a verb — and should I write...

Ah, ah, so... yes.

I think I'd keep it as a verb and then put it in this... A greeting.

Because banks are bad.

Ah, ok. Wait, wait, wait, wait. So, just around x,y,r to 2.

Convert x, y, r to the nearest integer using banker's rounding and increment element v, x, y, r by 1. Does this sound right in English? Well, I think ROUND is better than CONVERT. Ok, so I repeat... yes. So I continue...

But in this case, is the formal term "banker's"? Yes. Or should I write criteria, banking criteria? No.

Well, ok, I like that.

And there should be an apostrophe, shouldn't there? Banker's rounding — yes, that seems to be the most common term.

Have a nice day, darling.

Ah-ha! Now I understand why they use it.

Why do they use it?

Banker's rounding. To avoid. From.

It's because when you use discrete arrays that contain numbers 0, 0.1, and so on up to some certain number, then you get equal numbers rounding. Subtitles by the Amara.org community.

Eh?

But...

Maybe we should — if we want to be very precise — maybe I think say, round each element in... And then you have the X, Y, R, because that's actually what we're doing, element by element.

But maybe that's too detailed. We'll probably violate that elsewhere anyway, so keep it as it is now. It's much better, Esteban. Much clearer. Ok, so step 3. Select Pop N.

TIE is broken by first occurrence, but first occurrence requires you to specify how to iterate through the action.

I don't know how... I've just asked. I've used Python. Show me the... It's hidden in the Python code, so...

This kind of... I don't know if the topic is deep-level coding, but... when I used it, they gave me the 5 with the highest votes. Python gave me the results — I don't know about the ties.

Well, you can easily make a case where you can test it. You just make an array and put it in...

Or the other way is that ties will happen, so it has to be specified how ties are handled. So it's good that you mentioned it, because they will happen and they will happen surprisingly often. So we need to specify how — but then, by first occurrence, that means you understand that the occurrence requires you to specify what it is.

And how do you actually find the top peaks?

Because you have a three-dimensional array here, which means it could be an n³ iteration — so from a complexity perspective it's pretty bad if you actually do it in a cube. And there is almost certainly some smarter way of doing that search. And probably some smarter way is implemented in Python.

I assume so, but in this case n³ is... has typically been around... R is between... It has a range of about 20. X and Y, maybe 40. And that is... Not too large, eighty.

Yes, but for algorithms to do things it's common to specify their complexity.

Well, it's the size of V, actually — so what is the range of X, Y, and R values in V? That's actually what it becomes.

And they're different, so it's actually not cubic.

Ah, ok, so...

And actually, the X and the Y — how do you handle that for our infinite resolution case? Because for the others, X and Y is the resolution. I mean, X max, XY is the resolution. But what value do you put for the infinite resolution case?

Uff, I don't remember right now. No, no, I don't remember.

Stupid code right now.

Right now, B...

Wait...

You can make the signs of B for the different cases you calculated before.

A long time ago, this code has had many changes.

And from our moment until now — due to exactly what we're discussing — I tried to understand how to do it. V is a collections Counter.

How do I write here?

Did you see what I wrote? No, where did you write something? In the Meet chat.

OK, let me check the chat.

Ok, I don't know the Counter data type well enough. And being able to say how it's working — and the top peaks are just that line.

So, at the beginning it wasn't like this. This change is...

A long time ago I remember this is... But before you told me V was an Array name. I thought it was. But looking at this I remember that it was changed a long time ago because...

Why was it?

But now it's a map, and that map takes a pair of points comma.

Okay, I don't know the data types well enough to know what that actually implies.

You have to check what that means.

At least we need to change the n cells to elements.

Which line is that in this case? Line 13. You're talking about cells there again.

So CELLS goes to ELEMENT. Yes, ELEMENT. Let B be the element N. So B with the largest number of elements.

So I've seen a lot of cell code and...

This looks like the most robust, because yes, sometimes they're really vague. Yes, I know. Some pseudo-types are really horrible and I don't know why they've even been published. Let P be the N elements of V in order. In decreasing order — I think that's better than "with the maximum numbers."

Because that's actually what you mean here — in decreasing order.

Yes, let P be the n elements of B in decreasing order. Why? Or in decreasing, in the order of decreasing count.

Because the highest numbers — again, we don't know what order we have in P. So we want to imply that the first element in P is the one with the highest count, right? Right?

It's not important.

But isn't that what the Counter returns? Ah, uh... I don't know.

I don't know.

Ok, so I'm trying to... Ask Claude about the questions I've asked.

And then we ask Claude to give us a reference for each answer, so we can double-check them. Because Claude should be able to give a reference to its own book — through a manual so that it's easier to detect the correct answer.

That's faster than us speculating about things.

And our cell code is no longer a cell code — it begins being the definition of the algorithm. Yes, so this Counter V seems a lot like a... Dictionary.

It counts for each case that repeats.

It is — what did you say? Like a dictionary.

What are the properties of a dictionary?

It has a triplet, colon, and the repetition count.

So when asking for the first one — like the top N —

I suppose it stores results according to when they appeared first.

So we have a list of 5,000 triplet points, each of them. It creates X, Y, R and then V adds this according to when it appeared in the original order, which is random.

Ok, so...

If we want to write Zelda code, then we should say these things, actually. But if we want to write the definition of the algorithm...

And we don't consider it as... I say, mathematically...

This doesn't matter, except of course for how ties have been handled. But from the point of view of the implementation and the time complexity of all of this and how much memory is required, this actually matters. Because this means that you actually introduce a sparse object.

In fact, you avoid all the elements that are zero, that never occur. Yes.

And you know, the reason we now have all these problems with not really knowing why things sometimes change in the results is because we haven't defined the algorithm correctly — so that the description of the algorithm corresponds to what is implemented in the code — and then when you change things in the code

and suddenly we have a change in the results and we haven't — or you haven't — stayed attentive to what change produces what kind of change in the results. That's why we have reproducibility problems. So the only way to fix it is to be really...

To be really strict here? To be really strict here. So to write our algorithm definition, we write exactly what we want with the things. And then we ensure that the implemented code corresponds exactly to that and that there are no additional things in the code.

So we have a decision to make and it's that it should actually be here.

To make this machinery explicit.

So...

No, I really don't think so, because if someone wants to do this without a mapping, or wants to do it with an array...

The result should be the same — after running it 100 times it should be the same. I really don't think... because I'm using a collections tool that maps and creates a dictionary — that shouldn't change. If someone wants to do it another way, I don't think it's a problem.

Personally, and maybe it's bad, I don't understand why that would give bad results. It's not a question of giving bad results — it's a question of giving a different result. You have a very simple example of when it will give a different result. And that is a different order of random triplets, because that now

determines how ties are selected and then you have ties — sometimes one tie will be included, sometimes another. And that will immediately affect the result.

But if I'm running it one hundred times...

And if you take the mean of things, yes, then the mean shouldn't be affected. But if you run it one hundred times — it's not part of the algorithm. The algorithm runs once for each bridge.

The problem we're suffering from, Esteban, is that we cannot reproduce our own work. Or you can't reproduce your own work — that's been the state until now. So that's the thing we have to fix, because we don't want to have a review

And then again we're asked to make some changes to the algorithm to test something, and then again we have an explosion of... new results that are different and we don't know why or what has changed. Because that's what keeps us back.

So the way we solved it is that either we write the code so cleanly and so clearly that we can always diff the exact lines of code. And each line has a clear meaning. Or then we solve it by defining the algorithm on paper and making sure that the code corresponds to the algorithm.

Now, the code for us humans is still always a little harder to read, I think. That's why, personally, I prefer to define the algorithm.

Does the text say Algorithm 1? Yes.

Yeah, but I mean use of your map.

Now I understand why you called it map. I mean, for us to ensure the purpose of having predictable behavior and to detect changes, we should actually write it so that it reflects the code. So, Initialize. And then that map when it's initialized — yes it's empty — and then we add the triplets and the count of them. Why don't we write that: initialize an empty — and it's not integer vote array but it's an empty Vote MAP.

And then we specify that it's the platform count. So let's write it as the code. And then we get the row 11 is then increased the count. And we don't have any element or any things like this. Okay, so start now row four: Initialize an empty vote map.

Yes, I have to specify that V — the map V — is parentheses, the triplet, comma, count, end of parenthesis.

Yeah, but you don't write triplet and it's not colon — I think it's comma instead. You have to write the counter has. Okay, I can put comma but okay.

On the triplet, what should I put there? Like X, Y, R. Yes.

Shouldn't it be italic X, Y, and R — today really dear Roman? And then it seems like you have white space between V and the parentheses but you miss the white space after the comma before count. Now it's good? There's a white space missing between the comma and count. Here's something. All right.

So I added on both sides — you want only on the first side?

Only on the right side. Yeah. Now it should be easier to read. Um...

And then... Line 11.

Yes?

So round each element of. And then you have this X comma Y comma R. And increase the count. Oh. The X comma Y comma R blah blah blah.

You have to replace the v with the square brackets now with the v — the map notation from above. So you just copied the vote map from above and you insert it there instead of this v square bracket.

So I... I have a question now because the "count" here is italic and up above it's text. Above here?

Yeah. I think it is better to have it as text. Okay, so this one. Yeah, that's why I do it. The copy part was that. And actually instead of "count of" maybe we should say "increase the count in" because we have the term count mentioned there. There. Yeah, okay. Then line 13. For a map, do we normally call them elements then or what do we normally call them?

Well, I'm not sure either. Um...

Tell.

They... You seem to mainly be speaking about elements, actually.

I think yes. You prefer elements? Yeah.

Okay, good. Then step 4: localized cubes scoring around each peak. For each peak P. Bye. You need to introduce...

I don't think you need to mention "peak" there, the word peak. You can just say for each P_i in P do. Because you mentioned the peak already on the row before. Yes.

So here it's elements. Yeah.

Okay. What is this n or of the eye. The neighborhood.

So it's the cube. And when... So this tau somehow is the distance to the side. So when tau is 1, you have a 3×3×3. When tau is 2 you have a 5×5 cube.

I understand that, but you say it's cube half-width up there in the definition.

Yes. So it's not the width of the cube — it's somehow half of it. No, but it's not the half of it because the half of it would be 1.5.

Yes, I know. So...

Somehow it's... For me... It fulfills the criteria of the radius given in the infinite norm. Because always from the center — the infinite norm, if it's equal to 1, you get the cube. There's no coordinate that has a distance bigger than 1. Mmm.

But what do we benefit from defining tau like this?

Why don't we just define the cube side length? And then we have tau equal to 3. What's wrong with that? Nothing?

This is more like a radius thing. Because if we define it like that, somebody will say: For example, oh, I want a 4×4×4. Because...

Then people can do four times four if they want.

But it will not be centered.

Yes, I know it will not be centered and it will again depend on how exactly it's implemented — what it will become, how it will be centered. But I mean we avoid introducing something like "cube half-width," which is not really the radius and it's not really half the width either. Um... And then you call it "return the window." Why not — yeah, I mean just call it — I would say just use 3 there and then we make things simpler. Or do we make something else complicated? I see you used tau on the next line.

So what do you actually have there? You have T_i plus delta. So, I...

What happened?

I changed something and now it doesn't work. I want to be happy again. Why did I come back with that?

Why do you have the vote-weighted centroid here, when it seems like you are never using it? Wait, wait, okay. Return the vote-weighted centroid as the answer — is it? Yes?

That is correct.

What is it that you don't like? This?

Or this? Well, first and foremost they don't agree with the notation we have for the map. Aww. Here.

The C gap. Yes.

So, because, yes.

Why do we call it "sample size" — the n triplet flat, 5000 — that's not the sample size actually. The sample size is how many data points we have. So we shouldn't call that sample size. We should just call that either the triplet count or number of triplets. And why do you call that "center tolerance" when it's not actually used for the center? Wait. Okay, it's used to... So, the first thing you said: "Where are you reading?" I'm reading input line. So line 0. Input. Okay, here.

Okay. Yeah.

So just change sample size to "triplet count" for example — then we have "triplet count," "peak count," so they make sense. I updated the last part: cube side, kappa equals to 3. And then I updated the last part. But you also need to — you have the cube side length. Uh-huh. But I mean it doesn't work — the notation that you have from 16 to 18 — because it doesn't fit... It doesn't fit with the notation for the map.

Because of the C.

No, because of the square bracket C. There are no square brackets in the map. There is no place to insert anything. This? Yeah. Yes, exactly.

Yes, so how in the mapping — how do we extract the count? Yeah, that's exactly what I'm trying to think about.

How to write that in the most concise and easy to understand manner. And actually that goes back to the fact that we say P_i — the n elements. So P_i is actually... Circle. Follow me. A circle. Yes. P_i is X, Y, R. Yeah. Okay, so please look here. Yes? What if...

Is this too ugly?

Well, I think you should...

Put the equal signs outside. But yeah, that has the advantage of introducing P and C. So this part is math, this is text. Here, so... well, the C you would need to change to italic then also. So then it would be math: V(p, c) and then you put... Yeah, but you can on that line in — use it then. Wait, wait. What if... What up? Yeah, there's just some parentheses that are so. Instead of where you can use: Initialize an empty vote map, and then you remove the comma after it and then just write P equals da da da and C equals — and count C you can write, because C is not equal to count, it's count C. Initialize the loop. And we vote map V(P, C).

With P equal to that and what?

And count C. Not P equals count, but count C.

Yes. And then okay.

I will try to continue this. I will send you an Asana message when this is done. I know we have. Yes?

Yeah, we need to fix line 11 then also. And you just... I think in line 11 there is an advantage to have X, Y, R inserted. But you change the count to just C. Yeah.

Shouldn't I... If it's okay or should I change that to "and"?

So it becomes less clear that it's X, Y, R that is P. So, uh... And you have X, Y, R earlier that you speak about. So I think it's actually clearer to write it out there. Okay.

The transcribing part of Notion has two minutes left so I will stop it and send it to you.
