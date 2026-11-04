---
name: treasury-bulletins
description: How to find a figure in Treasury Bulletin text files without reading them whole.
triggers:
  - treasury
  - bulletin
---
# Finding figures in Treasury Bulletins

Each bulletin is several hundred kilobytes of text. Never print a whole file.

1. Pick the issue first. A figure for a period is usually published in an issue after that period ends, and later
   issues revise earlier numbers. If the question names an issue, use that one.
2. Search before reading: `grep -n -i "<table or row keyword>" corpus/<file>` finds the table, then read only the
   lines around the match (`sed -n 'START,ENDp' corpus/<file>`).
3. Read the table's header for its unit (millions, thousands, percent) and its period (fiscal or calendar year) before
   taking a number, and convert to the unit the question asks for.
4. Do arithmetic with a command (`python3 -c "..."`), not in your head.
5. Before answering, re-read the row and column you used and check they match every condition in the question.
