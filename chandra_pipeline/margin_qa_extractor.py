"""
Margin Q&A Extractor

Takes the text blocks Chandra found on the margin-only page (the question
numbers, boxed) and on the answers-only page. Each answer block is filed under
the question whose row it sits in.
"""

import re

import cv2

from chandra_config import MarginQaConfig


class MarginQAExtractor:

    def __init__(self, config: MarginQaConfig):

        self.question_pattern = (
            re.compile(config.question_pattern)
            if config.question_pattern is not None
            else None
        )
        self.answer_y_tolerance = config.answer_y_tolerance
        self.answer_separator = config.answer_separator
        self.box_color = tuple(config.box_color)
        self.box_thickness = config.box_thickness
        self.label_font_scale = config.label_font_scale
        self.label_thickness = config.label_thickness
        self.label_offset = config.label_offset

    # --------------------------------------------------

    def split(self, margin_blocks, answer_blocks):
        """Keeps the margin blocks that look like question numbers and drops empty
        ones. Both lists are in page coordinates. Returns (questions, answers),
        both in reading order (questions top to bottom)."""

        questions = [
            b for b in margin_blocks
            if b["text"] and (
                self.question_pattern is None
                or self.question_pattern.search(b["text"])
            )
        ]
        answers = [b for b in answer_blocks if b["text"]]

        questions.sort(key=lambda b: b["bbox"][1])
        answers.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

        return questions, answers

    # --------------------------------------------------

    def pair(self, questions, answers):
        """One row per question with the text of the answer blocks in its row.
        Answers above the first question go in a leading row with an empty question."""

        tops = [q["bbox"][1] - self.answer_y_tolerance for q in questions]

        grouped = [[] for _ in questions]
        orphans = []

        for answer in answers:

            x0, y0, x1, y1 = answer["bbox"]
            centre = (y0 + y1) / 2

            owner = None
            for i, top in enumerate(tops):
                if top <= centre:
                    owner = i

            if owner is None:
                orphans.append(answer)
            else:
                grouped[owner].append(answer)

        def join(items):
            return self.answer_separator.join(a["text"] for a in items)

        rows = []

        if orphans:
            rows.append({
                "question": "",
                "question_bbox": None,
                "answer": join(orphans),
            })

        for question, items in zip(questions, grouped):
            rows.append({
                "question": question["text"],
                "question_bbox": question["bbox"],
                "answer": join(items),
            })

        return rows

    # --------------------------------------------------

    def draw(self, page, questions):
        """Copy of `page` with only the left-margin question boxes drawn."""

        canvas = page.copy()

        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        for q in questions:

            x0, y0, x1, y1 = q["bbox"]

            cv2.rectangle(
                canvas, (x0, y0), (x1, y1),
                self.box_color, self.box_thickness,
            )

            cv2.putText(
                canvas, q["text"].replace("\n", " "),
                (x0, max(0, y0 - self.label_offset)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.label_font_scale, self.box_color, self.label_thickness,
            )

        return canvas
