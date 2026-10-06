# Question 3 input documents

Put the documents to process here (PNG, JPG/JPEG or PDF). `scripts/run_question3.py` and the
"Use the supplied sample documents" option in the app read this folder.

The supplied evaluation files (Aadhaar, PAN, driving licence, passport and insurance forms) are identity
documents, so they are **not committed to Git**: everything in this folder except this README is
git-ignored. Copy them here from the assignment pack:

```
Aadhar.png
ChatGPT Image May 2, 2026, 03_43_11 PM.png
ChatGPT Image May 2, 2026, 03_52_54 PM.png
ECS.jpeg
Fatca.jpeg
ID.png
Illustration.jpeg
Moral.jpeg
split.jpeg
suitability.jpeg
Proposal Ashok.pdf
Assignment Ashok.pdf
```

To include them in a **private** repository anyway: `git add -f data/documents && git commit`.
Never push them to a public repository without explicit permission.
