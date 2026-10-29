# Draft outreach email — **NOT SENT**

Nothing here has been sent and no laboratory has been contacted. Before sending, confirm the
recipient's current address from an official institutional directory, check that the publications
cited are theirs, and have a supervisor review the message.

A note on tone: academics receive a lot of speculative AI-and-drug-discovery mail. The draft
therefore leads with the limitation rather than the claim, keeps to one screen, asks only for a
conversation, and makes clear we want their measurement rather than their endorsement.

---

## Version A — Keillor group (best capability match)

**Subject:** Prospective test of a TG2/FXIIIa selectivity model — 16 compounds, would you be open to a conversation?

Dear Professor Keillor,

I am an MSc student working on evaluation methodology for machine-learned bioactivity prediction,
and I am writing to ask whether you would be willing to discuss a small collaboration. I have no
laboratory of my own, and the experiment I describe below needs exactly the capabilities your
group already has.

Using public data, I assembled 147 compounds with an exact IC50 against both human TG2 and human
FXIII-A measured *within the same publication*, so that each selectivity ratio is a within-study
comparison. A random forest predicts the continuous log selectivity ratio within one 69-compound
series (Prime et al., J Med Chem 2012) at Spearman 0.42 under a Butina cluster split, stable when
close analogs are removed from training. I should be equally clear that a second series looked
just as good under a scaffold split and collapsed to zero under a cluster split, and that
prediction fails entirely across the two series. The honest summary is one series, retrospective,
and a chemotype from a Huntington's programme rather than a celiac one.

That seems to me worth one prospective test rather than further modelling. The design is fixed in
advance: 16 untested analogs of that series, chosen to span the full predicted selectivity range
rather than the top of it, with success defined beforehand as Spearman ≥ 0.5 and failure below
0.3. The measurements needed are TG2 and FXIII-A dose–response run in parallel, k_inact/K_I for
confirmed actives, and a glutathione stability readout. Your group's work on acrylamide TG2
inhibitors, Factor XIIIa, and DTNB glutathione reactivity is the reason I am writing to you
specifically — and several of your papers are in the dataset I modelled, so your measurements are
part of what is being tested.

I would propose joint authorship and publication of the data and code whatever the outcome,
including a negative one. I have no commercial interest in any compound.

Would you be open to a short conversation about whether this is of interest? I can send a
two-page summary and the preregistered design, and I am happy to be told the idea needs more work
before it deserves bench time.

With thanks for your time,

Kareem Itani
MSc candidate, [programme and institution]
[email] · [supervisor name and contact]

---

## Version B — Khosla laboratory (disease-area match)

**Subject:** TG2/FXIIIa selectivity prediction — a question about dihydroisoxazole data

Dear Professor Khosla,

I am an MSc student working on evaluation methodology for machine-learned bioactivity prediction,
with a focus on TG2. I am writing with a narrow question rather than a proposal.

From public data I assembled 147 compounds with exact IC50 values against both human TG2 and human
FXIII-A measured within a single publication. Across that set, 8 compounds exceed 100-fold
selectivity for TG2, so selective chemistry against this antitarget clearly exists. A model
predicts the selectivity ratio within one acrylamide series at Spearman 0.42 under a cluster
split, but fails across series and collapsed to zero in a second series under the same test.

The limitation that prompts my question is that the series where the model works is from a
Huntington's programme, while the chemotype most relevant to celiac disease is the
3-bromo-4,5-dihydroisoxazole lineage from your 2014 J Med Chem paper. In public data that series
has only 8 compounds with paired FXIII-A measurements, which is too few to analyse. If further
TG2/TG1/TG3/FXIIIa data for that series exists, or if there is a reason I should not expect
selectivity to be learnable there, I would be grateful to know.

If it happened to be of interest, I would gladly discuss a small prospective test, but I am
conscious that my current result is in the wrong chemotype for your programme, and I am mainly
writing because you would know whether the question is worth pursuing at all.

With thanks for your time,

Kareem Itani
MSc candidate, [programme and institution]
[email] · [supervisor name and contact]

---

## Before sending — checklist

- [ ] Supervisor has reviewed the message and agreed to be named
- [ ] Recipient's email confirmed from an official institutional directory, not inferred
- [ ] Cited publications confirmed to be the recipient's
- [ ] `research_summary.md` attached or offered, not pasted into the body
- [ ] No candidate compounds named, and no claim of therapeutic relevance
- [ ] Placeholders (`[programme and institution]`, `[email]`, supervisor details) filled in
- [ ] One recipient at a time; not a mass mailing
