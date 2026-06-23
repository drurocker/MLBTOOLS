# MLB DFS Stack Command Center v4

Streamlit app for ranking MLB DFS stacks using:

1. Scoring % Sheet as the slate master
2. Matchups Master for trends
3. Park Factors for run/HR environment
4. DK Salaries for game/home park/player-team map
5. Optional Portfolio CSV for primary/secondary stack detection and lineup grading

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Upload order

- Scoring % Sheet
- Matchups Master Sheet
- Park Factors Sheet
- DK Salaries Sheet
- Optional Lineup Portfolio Export

The app only ranks teams from the scoring sheet.
