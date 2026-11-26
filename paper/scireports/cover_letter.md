# Cover letter (draft)

[TO BE FILLED: date]

To the Editor, *Scientific Reports*,

We submit the manuscript **"Observability-gated fusion of sensor measurements and a neural network improves learned visual–inertial odometry"** (Nada El Madah, Magdy Zakaria, Samir Elmougy, Shahenda Sarhan, Mohamed Handosa and Rasha Elsayed Sakr) for consideration as an Article in *Scientific Reports*.

**What the paper does.** Robots, drones and headsets that move where satellite positioning is unavailable can track their motion with a camera and an inertial sensor. Recent systems train one neural network to predict the whole motion, although the sensors already measure much of it. We introduce a simple diagnostic, an *oracle swap*, which replaces one predicted component of the motion at a time by ground truth to show which component the network gets wrong. The diagnosis differs between datasets (rotation and step length on a driving dataset, direction of travel on a drone dataset) and uncovered a previously unreported 4% timing error in inertial files distributed with several released models. We then propose an observability gate that uses a sensor measurement where the sensors can observe a component and a frozen network elsewhere. With one configuration, fixed before the test runs, it lowers the translation error of a published model on KITTI from 1.96% to 1.56% and the error of released models from three further families by 45–51%.

**Why *Scientific Reports*.** The journal publishes technically sound work across the sciences and engineering, and the contribution here is methodological and diagnostic rather than a new state of the art. We say so explicitly: a classical system remains about 22 times more accurate on one of the datasets, and we report every failure, including a pre-registered version that failed on one dataset and a post-hoc revision that was no better on a sequence it had not seen. We believe this transparent reporting, together with the oracle-swap procedure that readers in any field using learned sensor fusion can apply, suits the journal's emphasis on rigour over novelty, and the paper is written for readers from many disciplines (navigation, robotics, sensing, machine learning).

**Corresponding author.** Nada El Madah, Department of Computer Science, Faculty of Computers and Information, Mansoura University, Mansoura, Egypt; email nadaelmadah@mans.edu.eg; ORCID 0000-0002-2537-2746. Affiliation of all authors: Computer Science, Mansoura University, Mansoura, Egypt.

**Prior discussion with a *Scientific Reports* Editor.** [TO BE FILLED: none / name of the Editor and date].

**Suggested reviewers.** [TO BE FILLED: names, affiliations and emails; none supplied yet.]

**Excluded referees.** [TO BE FILLED: names and reasons, or none.]

**Declarations.** The manuscript is not under consideration elsewhere and has not been published previously. All authors have approved the submission. The authors declare no competing interests. [TO BE FILLED: use of large language models.] Data and code availability statements are in the manuscript [TO BE FILLED: repository and DOI].

Sincerely,

Nada El Madah (corresponding author)
