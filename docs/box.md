<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Box questions: where is it in the picture

A Box question finds where something is in an image and answers with one box.
Use it to crop the part of a photo that matters, to point a person at it, or to check that a thing sits where the state says it does.
It runs on a Gemma server started with `--images on`; the wire is in [api.md](api.md#box-questions).

## Asking

```json
{"state": "The listing photo: PHOTO",
 "questions": {"mug": {"type": "box", "instructions": "the blue mug"}},
 "assets": {"PHOTO": {"type": "image", "format": "png", "data_base64": "..."}}}
```

- Say what to find in `instructions`: a short noun phrase ("the red car on the left", "the price tag").
  Without instructions the question finds the main subject.
- Say which image when there are several, by the name the state gives it: "the red star in scene_b.png".
  Put a label before each image in the state so the names are in the text the model reads.
- Give the target as a picture by putting the image in the question: `"the shape in SCENE that matches the shape in REFERENCE"`, with both as assets.
  An image in a question's instructions belongs to that question and is not seen by the others.
- Ask one thing per question.
  A Box answer is one box; to find three mugs, ask for "the left mug", "the middle mug" and "the right mug", in one request.

## Reading the answer

```json
{"mug": {"type": "box", "box": {"top": 0.12, "bottom": 0.58, "left": 0.31, "right": 0.66}, "confidence": 0.93}}
```

The edges are fractions of the image you sent: multiply `left` and `right` by its width and `top` and `bottom` by its height to crop.

```python
x0, x1 = round(b["left"] * width), round(b["right"] * width)
y0, y1 = round(b["top"] * height), round(b["bottom"] * height)
crop = image.crop((x0, y0, x1, y1))
```

The edges are an expected value over the digits the model considered, not only its first guess, so a box the model was unsure about sits between its candidates.

## Confidence, and "is it there"

A Box question always answers with a box, because the reply is one box by construction.
When the target may not be in the picture, the answer's `confidence` is the signal: it is the probability that the target is in the picture, from a yes/no read of the same image on the same prefix, times how sure the model is of the box's edges.
Pair the Box with a noul question for a decision your code makes ("is there a mug?") and use the Box only where that answers yes:

```json
{"has_mug": {"type": "noul", "instructions": "Is there a mug in the picture?"},
 "mug": {"type": "box", "instructions": "the mug"}}
```

Until a calibration for the served model is loaded (`--box-confidence FILE`, or `box_confidence.json` in the model's folder), `confidence` is that product and not a measured rate: use it to rank and to compare, not as a probability of a correct box.

## Large images

An image is read at a fixed number of pieces (`--image-budget`), so a large image gives each object fewer.
A small object in a large image is found better when you crop the image to the region first and ask again in the crop; the fractions then refer to the crop.
A higher `--image-budget` (560) spends more tokens per image and helps in the same way.

## Limits

- One box per question, never none.
- Not served with two orders, abstention, a task, or a request that asks for `"layout": "header"`; the server needs `--images on` on a Gemma base (not `--image-model`, not the MLX backend).
- A Box question is always read under the plain layout, whatever the server's default, so a request without `"layout"` is answered on a Gemma base too.
- A request that mixes a Box with other questions on a server whose layout is the header one reads the state twice: once for the other questions under the header layout, and once for the Box questions and their presence reads under the plain layout.
  The `x-decisio-prompt` header reports the layout of the other questions, and `timing.box.layout` says `plain`.
- A Box question writes up to 32 tokens, so it costs more than a noul or choice question; several Box questions of one request run as one batch on the cached prefix of the state.
- How accurate the boxes are, how well the confidence tracks them and what a Box costs in time are not measured yet; this page will carry the figures when they are.
