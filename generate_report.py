import subprocess
import os

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
TEX_FILE = os.path.join(REPORT_DIR, "Cooper_Morgan_Lab7.tex")
PDF_FILE = os.path.join(REPORT_DIR, "Cooper_Morgan_Lab7.pdf")

# TODO: update filenames once figures are generated for Lab 7
# Lab 7 calls for visualizations of: (1) cumulative reward over real time steps,
# (2) episodes-until-optimal, (3) sample efficiency, plus Dyna-Q+ adaptation curve
# and prioritized-sweeping comparison. Pick 2-4 of the most insightful for the PDF.
FIGURE_1 = "figures/placeholder_1.png"
FIGURE_2 = "figures/placeholder_2.png"
FIGURE_3 = "figures/placeholder_3.png"
FIGURE_4 = "figures/placeholder_4.png"

tex_content = r"""
\documentclass[12pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{parskip}
\usepackage{float}

\titleformat{\section}{\large\bfseries}{}{0em}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}

\title{Lab 7: Planning and Learning Integration}
\author{Morgan Cooper \\ MSDS 684 --- Reinforcement Learning}
\date{\today}

\begin{document}
\maketitle
\newpage

\section{Section 1: Project Overview}

% Section 1 target: 400-500 words. Weight: 25 points (25%).
% Explain the WHAT and WHY of the lab. This is a conceptual overview, NOT a methods
% section, NOT a code walkthrough, NOT a results summary.
%
% Required content (per Lab Directions):
%   - Problem/Question: What RL problem are you investigating?
%       -> For Lab 7: how does integrating a learned model with direct RL (Dyna-Q)
%          change sample efficiency vs. pure model-free Q-learning on Taxi-v3?
%   - Core Concepts: which Sutton & Barto concepts are you exploring?
%       -> Model-based vs. model-free RL, the Dyna architecture, Dyna-Q+ exploration
%          bonus, prioritized sweeping (S&B Chapter 8).
%   - Theoretical Grounding: connect to S&B Ch. 8 readings.
%   - Environment Description: state space, action space, rewards, terminations.
%       -> Taxi-v3 (Gymnasium): 500 discrete states, 6 discrete actions, reward
%          structure (-1 per step, +20 dropoff, -10 illegal pickup/dropoff),
%          terminates on successful dropoff.
%       -> Plus the dynamic-change wrapper applied after 1000 steps for the
%          Dyna-Q+ comparison.
%   - Expected Behavior: hypothesize what will happen and why.
%       -> e.g. Dyna-Q with larger n converges in fewer real env steps; Dyna-Q+
%          recovers faster than Dyna-Q after the environment shift; prioritized
%          sweeping beats uniform planning early on.

% TODO: write 400-500 words covering the bullets above.

\newpage
\section{Section 2: Deliverables}

% Section 2 target: weighted 35 points (35%). Combines implementation summary,
% results, and analysis. Hard rules: NO raw code listings, NO console output dumps.
% Code lives in GitHub; analysis lives here.

\subsection{GitHub Repository}
% Place the repo URL prominently at the top of this section.
\begin{verbatim}
GitHub Repository: https://github.com/<user>/<repo>/tree/main/lab7
\end{verbatim}

\subsection{Implementation Summary}

% 100-150 words of brief prose. Cover:
%   - What you implemented: Dyna-Q (Direct RL + Model Learning + Planning),
%     Dyna-Q+, and prioritized sweeping on Taxi-v3.
%   - Experimental setup: e.g. "X episodes, Y seeds, n in {0, 5, 10, 50}".
%   - Key hyperparameters chosen: alpha, gamma, epsilon, kappa (Dyna-Q+ bonus),
%     priority threshold (prioritized sweeping).
%   - NOT pseudocode, NOT line-by-line.

% TODO: write the 100-150 word implementation summary.

\subsection{Key Results \& Analysis}

% 400-600 words plus 2-4 visualizations.
% Discussion must address:
%   - What do the results show about algorithm behavior?
%   - How do they relate to S&B Ch. 8 (cite chapters/sections)?
%   - What didn't work as expected? Why?
%   - How did hyperparameters (n planning steps, kappa, priority threshold) affect
%     performance?
%   - What does this teach you about model-based RL?
%
% Captions must be INTERPRETIVE, not just descriptive. Each caption should explain
% what the figure shows AND why it matters / what it implies about the algorithm.

% TODO: write the analysis prose. Suggested structure for Lab 7:
%   1. Sample efficiency: pure Q-learning (n=0) vs. Dyna-Q (n in {5,10,50}).
%      Reference Figure 1 (cumulative reward over real time steps).
%   2. Episodes-until-optimal and real-env interactions saved by planning.
%      Reference Figure 2.
%   3. Dyna-Q+ vs. Dyna-Q after the 1000-step environment change.
%      Reference Figure 3 -- show the adaptation gap and tie back to
%      kappa*sqrt(tau) bonus from S&B Section 8.3.
%   4. Prioritized sweeping vs. uniform random planning.
%      Reference Figure 4 -- discuss why focusing updates on high-TD-error pairs
%      accelerates propagation (S&B Section 8.4).

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{""" + FIGURE_1 + r"""}
% TODO: replace with an interpretive caption. Example structure:
% "Figure 1: <what the curves show> because <mechanism>. <Quantitative takeaway>
%  demonstrating <conceptual point tied to S&B Ch. 8>."
\caption{TODO: interpretive caption for cumulative reward over real time steps,
comparing pure Q-learning (n=0) and Dyna-Q with n $\in \{5, 10, 50\}$.}
\label{fig:learning_curves}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{""" + FIGURE_2 + r"""}
\caption{TODO: interpretive caption for episodes-until-optimal-performance
and/or real-environment interactions required across planning-step settings.}
\label{fig:sample_efficiency}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{""" + FIGURE_3 + r"""}
\caption{TODO: interpretive caption for Dyna-Q vs. Dyna-Q+ after the 1000-step
environment change; discuss how the $\kappa\sqrt{\tau}$ exploration bonus drives
faster recovery.}
\label{fig:dyna_q_plus}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{""" + FIGURE_4 + r"""}
\caption{TODO: interpretive caption for prioritized sweeping vs. uniform random
planning; tie to TD-error magnitude as a focus heuristic.}
\label{fig:prioritized_sweeping}
\end{figure}

\section{Section 3: AI Use Reflection}

% Section 3 target: 250-350 words. Weight: 35 points (35%) -- the most heavily
% scored section. Document the iteration process, not just the final result.

\subsection{Initial Interaction}

% 50-75 words.
%   - What did you ask the AI to help with?
%   - What was your initial prompt?
%   - What code/explanation did it provide?

% TODO: write the initial-interaction paragraph for Lab 7.

\subsection{Iteration Cycle}

% 150-200 words. THE most important subsection.
% Describe at least 2-3 concrete debugging cycles. For each:
%   - The error/problem you encountered
%   - Your follow-up prompt to AI
%   - AI's response
%   - Whether it worked or needed more iteration
%
% Lab 7 likely candidates for genuine bugs to document:
%   - Dictionary-based model returning stale (r, s') after the env change
%     (motivates Dyna-Q+ and the kappa*sqrt(tau) bonus)
%   - Off-by-one in time_since[(s,a)] tracking
%   - heapq comparison errors when tuples have equal priority
%   - Forgetting to reset/decay model entries when wrapper changes the env
%   - Q-learning update on simulated experience using stale next-state values

\textbf{Iteration 1: TODO title}
% TODO: describe the error, the prompt, the AI response, and whether it worked.

\textbf{Iteration 2: TODO title}
% TODO: same structure as above.

\textbf{Iteration 3: TODO title}
% TODO: same structure as above.

\subsection{Critical Evaluation}

% 50-75 words.
%   - Did you catch any AI mistakes?
%   - Did you test alternative approaches?
%   - How did you verify the final solution was correct?

% TODO: write the critical-evaluation paragraph.

\subsection{Learning Reflection}

% 50-75 words.
%   - What did you learn about the RL algorithm through debugging?
%     (For Lab 7: model-based planning, exploration under non-stationarity,
%      prioritized sweeping)
%   - What did you learn about working with AI tools?
%   - What would you do differently next time?

% TODO: write the learning-reflection paragraph.


\section{Section 4: Speaker Notes}

% Section 4 target: ~5 minutes of presentation, 5-7 bullets. Weight: 10 points (10%).
% Cover: problem & motivation, method & key algorithmic choices, an important
% design decision or challenge, main result, key insight, and (optional) connection
% to future weeks or real-world applications.
% Format must be bullet points, not paragraphs.

\begin{itemize}
  \item \textbf{Problem:} TODO -- one-line problem statement and motivation
        (planning + learning integration on Taxi-v3).
  \item \textbf{Method:} TODO -- Dyna-Q with n planning steps; Dyna-Q+ exploration
        bonus; prioritized sweeping via heapq.
  \item \textbf{Design choice:} TODO -- e.g. dictionary-backed deterministic model,
        kappa value for the exploration bonus, priority threshold for sweeping.
  \item \textbf{Key result:} TODO -- e.g. n=50 reaches optimal in <X> real steps vs.
        <Y> for pure Q-learning; Dyna-Q+ recovers in <Z> steps after env change.
  \item \textbf{Insight:} TODO -- when models help (small/deterministic, expensive
        real env) vs. when they hurt (model error, non-stationarity).
  \item \textbf{Challenge:} TODO -- the most interesting bug from Section 3.
  \item \textbf{Connection:} TODO -- bridge to deep model-based RL (e.g. world
        models, MuZero) and the Lab 8 / final synthesis.
\end{itemize}

\section{References}

\begin{enumerate}
  \item Sutton, R. S., \& Barto, A. G. (2018). \textit{Reinforcement learning: An introduction} (2nd ed.). MIT Press.
  \item Anthropic. (2025). Claude Code [Large language model CLI tool]. \texttt{https://claude.ai}
  \item OpenAI. (2025). ChatGPT [Large language model]. \texttt{https://chat.openai.com}
\end{enumerate}

\end{document}
"""

def main():

    # Write temporary .tex file
    with open(TEX_FILE, "w") as f:
        f.write(tex_content)

    # Compile to PDF (run twice to resolve cross-references)
    for pass_num in (1, 2):
        print(f"Compiling to PDF (pass {pass_num})...")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", TEX_FILE],
            cwd=REPORT_DIR,
            capture_output=True,
            text=True,
        )

    if result.returncode == 0:
        print(f"PDF generated: {PDF_FILE}")
    else:
        print("pdflatex encountered issues:")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    # Clean up all LaTeX artifacts (keep only the PDF)
    for ext in [".tex", ".aux", ".log", ".out"]:
        artifact = os.path.join(REPORT_DIR, f"Cooper_Morgan_Lab7{ext}")
        if os.path.exists(artifact):
            os.remove(artifact)


if __name__ == "__main__":
    main()
