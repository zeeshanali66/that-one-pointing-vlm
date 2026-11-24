# Research log: "Bring me that one"

### 1. Why this study
Pointing plus "that one" (deixis) is how people refer to objects when talking to robots and AR assistants. VLM
spatial benchmarks mostly test words ("left of", "behind"), not gestures. Our earlier studies found that LLMs fail when
they must transform geometry themselves (rotate a viewpoint) and succeed when the system does it for them; pointing
is a natural next test of that idea, now with images.

### 2. Scene generator hung on the hardest setting
The first generator placed the rival object at the same distance as the target with a small random change. At 8
degrees and 2.5 m the two objects were then always closer than 0.75 m, overlapped, and the generator looped forever.
Fix: offset the rival in depth (0.4 to 0.8 m nearer or farther), the same rule for every setting.

### 3. A self-test that was wrong
The self-test checked that the target letter did not appear in the text prompt, but "A room..." contains the letter
A. Replaced with a check that the prompt never names the answer, and kept the geometric checks (truth = intended
target; rival exactly at the intended angle) for all 96 scenes.

### 4. The arm was too short to see
In the first render the body circle covered half the arm. Made the arm longer (1.3 m) and the body smaller before any
model call.

### 5. Invalid answers
Qwen 3.5 gave 14 invalid answers in the image condition (letters not in the scene, or "p" for person). They count as
wrong; reported separately.

### 6. Is the model following direction or proximity?
Raw accuracy alone could not say *why* models fail, so every answer is compared with two shortcuts (object nearest the
hand, nearest the body). Splitting accuracy by whether the target is also the nearest object showed that image-only
answers follow proximity (Gemma 4: 0.82 vs 0.23).

## Next steps
First-person (headset-view) renders, photos of real pointing, the arm direction given as an angle instead of a drawn
line, and a fine-tuned model on synthetic pointing scenes.
