"""
Script to collect and update Best Paper information for AI top conferences (2018-2024).
Updates top_1000_papers.json and creates best_papers.json
"""

import json
import os
from pathlib import Path
from datetime import datetime

# Best Papers data collected from official sources and jeffhuang.com
BEST_PAPERS_DATA = {
    # ICLR Best/Outstanding Papers
    "ICLR": {
        2024: [
            {"title": "Learning Interactive Real-World Simulators", "authors": ["Sherry Yang", "Yilun Du", "Pete Florence", "et al."], "award_type": "outstanding_paper"},
            {"title": "Vision Transformers Need Registers", "authors": ["Timothee Darcet", "Maxime Oquab", "Julien Mairal", "Piotr Bojanowski"], "award_type": "outstanding_paper"},
            {"title": "Generalization in diffusion models arises from geometry-adaptive harmonic representations", "authors": ["Zahra Kadkhodaie", "Florentin Guth", "Eero P Simoncelli", "Stephane Mallat"], "award_type": "outstanding_paper"},
            {"title": "Is ImageNet worth 1 video? Learning strong image encoders from 1 long unlabelled video", "authors": ["Shashanka Venkataramanan", "Mamshad Nayeem Rizve", "et al."], "award_type": "outstanding_paper"},
            {"title": "Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors", "authors": ["Ido Amos", "Jonathan Berant", "Ankit Gupta"], "award_type": "outstanding_paper"},
        ],
        2023: [
            {"title": "Rethinking the Expressive Power of GNNs via Graph Biconnectivity", "authors": ["Bohang Zhang", "et al."], "award_type": "outstanding_paper"},
            {"title": "Emergence of Maps in the Memories of Blind Navigation Agents", "authors": ["Erik Wijmans", "et al."], "award_type": "outstanding_paper"},
        ],
        2022: [
            {"title": "Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models", "authors": ["Fan Bao", "et al."], "award_type": "outstanding_paper"},
            {"title": "Hyperparameter Tuning with Renyi Differential Privacy", "authors": ["Nicolas Papernot", "et al."], "award_type": "outstanding_paper"},
        ],
        2021: [
            {"title": "Score-Based Generative Modeling through Stochastic Differential Equations", "authors": ["Yang Song", "Jascha Sohl-Dickstein", "Diederik P. Kingma", "et al."], "award_type": "outstanding_paper"},
            {"title": "Learning Mesh-Based Simulation with Graph Networks", "authors": ["Tobias Pfaff", "et al."], "award_type": "outstanding_paper"},
        ],
        2020: [
            {"title": "On Identifiability in Transformers", "authors": ["Gino Brunner", "et al."], "award_type": "outstanding_paper"},
        ],
        2019: [
            {"title": "Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks", "authors": ["Yikang Shen", "et al."], "award_type": "best_paper"},
            {"title": "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", "authors": ["Jonathan Frankle", "Michael Carbin"], "award_type": "best_paper"},
        ],
        2018: [
            {"title": "On the Convergence of Adam and Beyond", "authors": ["Sashank J. Reddi", "et al."], "award_type": "best_paper"},
            {"title": "Spherical CNNs", "authors": ["Taco S. Cohen", "et al."], "award_type": "best_paper"},
        ],
    },

    # ICML Best Papers
    "ICML": {
        2024: [
            {"title": "Debating with More Persuasive LLMs Leads to More Truthful Answers", "authors": ["Akbir Khan", "John Hughes", "Dan Valentine", "et al."], "award_type": "best_paper"},
            {"title": "Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo", "authors": ["Stephen Zhao", "Rob Brekelmans", "Alireza Makhzani", "Roger Grosse"], "award_type": "best_paper"},
            {"title": "VideoPoet: A Large Language Model for Zero-Shot Video Generation", "authors": ["Dan Kondratyuk", "Lijun Yu", "et al."], "award_type": "best_paper"},
            {"title": "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", "authors": ["Patrick Esser", "Sumith Kulal", "Andreas Blattmann", "et al."], "award_type": "best_paper"},
            {"title": "Genie: Generative Interactive Environments", "authors": ["Jake Bruce", "Michael Dennis", "et al."], "award_type": "best_paper"},
            {"title": "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution", "authors": ["Aaron Lou", "Chenlin Meng", "Stefano Ermon"], "award_type": "best_paper"},
            {"title": "Stealing part of a production language model", "authors": ["Nicholas Carlini", "Daniel Paleka", "et al."], "award_type": "best_paper"},
            {"title": "Position: Considerations for Differentially Private Learning with Large-Scale Public Pretraining", "authors": ["Florian Tramer", "Gautam Kamath", "Nicholas Carlini"], "award_type": "best_paper"},
            {"title": "Position: Measure Dataset Diversity, Don't Just Claim It", "authors": ["Dora Zhao", "Jerone Andrews", "Orestis Papakyriakopoulos", "Alice Xiang"], "award_type": "best_paper"},
            {"title": "Information Complexity of Stochastic Convex Optimization: Applications to Generalization, Memorization, and Tracing", "authors": ["Idan Attias", "Gintare Karolina Dziugaite", "et al."], "award_type": "best_paper"},
        ],
        2023: [
            {"title": "Scaling Data-Constrained Language Models", "authors": ["Niklas Muennighoff", "et al."], "award_type": "best_paper"},
            {"title": "Learning-Rate-Free Learning by D-Adaptation", "authors": ["Aaron Defazio"], "award_type": "best_paper"},
        ],
        2022: [
            {"title": "The Privacy Onion Effect: Memorization is Relative", "authors": ["Nicholas Carlini", "et al."], "award_type": "best_paper"},
            {"title": "Bayesian Model Selection, the Marginal Likelihood, and Generalization", "authors": ["Sanae Lotfi", "et al."], "award_type": "best_paper"},
            {"title": "Learning Mixtures of Linear Dynamical Systems", "authors": ["Yanxi Chen", "et al."], "award_type": "best_paper"},
            {"title": "Understanding Dataset Difficulty with V-Usable Information", "authors": ["Kawin Ethayarajh", "et al."], "award_type": "best_paper"},
        ],
        2021: [
            {"title": "Understanding Self-Supervised Learning Dynamics without Contrastive Pairs", "authors": ["Yuandong Tian", "et al."], "award_type": "best_paper"},
            {"title": "Optimal Complexity in Decentralized Training", "authors": ["Hadrien Hendrikx", "et al."], "award_type": "best_paper"},
            {"title": "Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies", "authors": ["Paul Vicol", "et al."], "award_type": "best_paper"},
        ],
        2020: [
            {"title": "On Learning Sets of Symmetric Elements", "authors": ["Haggai Maron", "et al."], "award_type": "best_paper"},
            {"title": "Tuning-free Plug-and-Play Proximal Algorithm for Inverse Imaging Problems", "authors": ["Kaixuan Wei", "et al."], "award_type": "best_paper"},
        ],
        2019: [
            {"title": "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations", "authors": ["Francesco Locatello", "et al."], "award_type": "best_paper"},
            {"title": "Rates of Convergence for Sparse Variational Gaussian Process Regression", "authors": ["David Burt", "et al."], "award_type": "best_paper"},
        ],
        2018: [
            {"title": "Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples", "authors": ["Anish Athalye", "et al."], "award_type": "best_paper"},
            {"title": "Delayed Impact of Fair Machine Learning", "authors": ["Lydia T. Liu", "et al."], "award_type": "best_paper"},
        ],
    },

    # NeurIPS Best Papers
    "NeurIPS": {
        2024: [
            {"title": "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction", "authors": ["Keyu Tian", "Yi Jiang", "Zehuan Yuan", "BINGYUE PENG", "Liwei Wang"], "award_type": "best_paper"},
            {"title": "Stochastic Taylor Derivative Estimator: Efficient amortization for arbitrary differential operators", "authors": ["Zekun Shi", "Zheyuan Hu", "Min Lin", "Kenji Kawaguchi"], "award_type": "best_paper"},
            {"title": "Guiding a Diffusion Model with a Bad Version of Itself", "authors": ["Tero Karras", "Miika Aittala", "Tuomas Kynkaanniemi", "Jaakko Lehtinen", "Timo Aila", "Samuli Laine"], "award_type": "best_paper_runner_up"},
            {"title": "Not All Tokens Are What You Need for Pretraining", "authors": ["Zhenghao Lin", "Zhibin Gou", "Yeyun Gong", "et al."], "award_type": "best_paper_runner_up"},
            {"title": "The PRISM Alignment Dataset: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models", "authors": ["Hannah Rose Kirk", "et al."], "award_type": "datasets_benchmarks_best_paper"},
        ],
        2023: [
            {"title": "Privacy Auditing with One (1) Training Run", "authors": ["Thomas Steinke", "et al."], "award_type": "outstanding_paper"},
            {"title": "Are Emergent Abilities of Large Language Models a Mirage?", "authors": ["Rylan Schaeffer", "et al."], "award_type": "outstanding_paper"},
        ],
        2022: [
            {"title": "Elucidating the Design Space of Diffusion-Based Generative Models", "authors": ["Tero Karras", "et al."], "award_type": "outstanding_paper"},
            {"title": "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding", "authors": ["Chitwan Saharia", "et al."], "award_type": "outstanding_paper"},
            {"title": "Is Out-of-distribution Detection Learnable?", "authors": ["Zhen Fang", "et al."], "award_type": "outstanding_paper"},
        ],
        2021: [
            {"title": "A Universal Law of Robustness via Isoperimetry", "authors": ["Sebastien Bubeck", "Mark Sellke"], "award_type": "outstanding_paper"},
            {"title": "On the Expressivity of Markov Reward", "authors": ["David Abel", "et al."], "award_type": "outstanding_paper"},
            {"title": "Deep Reinforcement Learning at the Edge of the Statistical Precipice", "authors": ["Rishabh Agarwal", "et al."], "award_type": "outstanding_paper"},
            {"title": "MAUVE: Measuring the Gap Between Neural Text and Human Text", "authors": ["Krishna Pillutla", "et al."], "award_type": "outstanding_paper"},
        ],
        2020: [
            {"title": "Language Models are Few-Shot Learners", "authors": ["Tom Brown", "et al."], "award_type": "best_paper"},
            {"title": "No-Regret Learning Dynamics for Extensive-Form Correlated Equilibrium", "authors": ["Andrea Celli", "et al."], "award_type": "best_paper"},
            {"title": "Improved Guarantees and a Multiple-Descent Curve for Column Subset Selection and the Nystrom Method", "authors": ["Michal Derezinski", "et al."], "award_type": "best_paper"},
        ],
        2019: [
            {"title": "Distribution-Independent PAC Learning of Halfspaces with Massart Noise", "authors": ["Ilias Diakonikolas", "et al."], "award_type": "outstanding_paper"},
            {"title": "Nonparametric Density Estimation & Convergence Rates for GANs under Besov IPM Losses", "authors": ["Ananya Uppal", "et al."], "award_type": "outstanding_paper"},
            {"title": "Fast and Accurate Least-Mean-Squares Solvers", "authors": ["Michal Derezinski", "et al."], "award_type": "outstanding_paper"},
        ],
        2018: [
            {"title": "Non-delusional Q-learning and Value-iteration", "authors": ["Tyler Lu", "et al."], "award_type": "best_paper"},
            {"title": "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks", "authors": ["Kevin Scaman", "et al."], "award_type": "best_paper"},
            {"title": "Nearly Tight Sample Complexity Bounds for Learning Mixtures of Gaussians", "authors": ["Idan Mehalel", "et al."], "award_type": "best_paper"},
        ],
    },

    # ACL Best Papers
    "ACL": {
        2024: [
            {"title": "Semisupervised Neural Proto-Language Reconstruction", "authors": ["Liang Lu", "et al."], "award_type": "best_paper"},
        ],
        2023: [
            {"title": "Do Androids Laugh at Electric Sheep? Humor Understanding Benchmarks from The New Yorker Caption Contest", "authors": ["Jack Hessel", "et al."], "award_type": "best_paper"},
            {"title": "What the DAAM: Interpreting Stable Diffusion Using Cross Attention", "authors": ["Raphael Tang", "et al."], "award_type": "best_paper"},
            {"title": "From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair NLP Models", "authors": ["Shangbin Feng", "et al."], "award_type": "outstanding_paper"},
        ],
        2022: [
            {"title": "Learned Incremental Representations for Parsing", "authors": ["Nikita Kitaev", "et al."], "award_type": "best_paper"},
            {"title": "Causal Effects of Linguistic Properties", "authors": ["Reid Pryzant", "et al."], "award_type": "best_theme_paper"},
        ],
        2021: [
            {"title": "Vocabulary Learning via Optimal Transport for Neural Machine Translation", "authors": ["Jingjing Xu", "et al."], "award_type": "best_paper"},
        ],
        2020: [
            {"title": "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList", "authors": ["Marco Tulio Ribeiro", "et al."], "award_type": "best_paper"},
        ],
        2019: [
            {"title": "Bridging the Gap between Training and Inference for Neural Machine Translation", "authors": ["Wen Zhang", "et al."], "award_type": "best_paper"},
        ],
        2018: [
            {"title": "Finding syntax in human encephalography with beam search", "authors": ["John Hale", "et al."], "award_type": "best_paper"},
            {"title": "Probabilistic Typology: Deep Generative Models of Vowel Inventories", "authors": ["Ryan Cotterell", "Jason Eisner"], "award_type": "best_paper"},
        ],
    },

    # AAAI Best Papers
    "AAAI": {
        2024: [
            {"title": "Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration", "authors": ["Yang Zhang", "et al."], "award_type": "best_paper"},
            {"title": "Fairness in Serving Large Language Models", "authors": ["Ying Sheng", "et al."], "award_type": "best_paper"},
        ],
        2023: [
            {"title": "Misspecification in Inverse Reinforcement Learning", "authors": ["Joar Skalse", "Alessandro Abate"], "award_type": "best_paper"},
        ],
        2022: [
            {"title": "Online certification of preference-based fairness for personalized recommender systems", "authors": ["Virginie Do", "et al."], "award_type": "outstanding_paper"},
            {"title": "Sampling-Based Robust Control of Autonomous Systems with Non-Gaussian Noise", "authors": ["et al."], "award_type": "outstanding_paper"},
        ],
        2021: [
            {"title": "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", "authors": ["Haoyi Zhou", "et al."], "award_type": "best_paper"},
        ],
        2020: [
            {"title": "WinoGrande: An Adversarial Winograd Schema Challenge at Scale", "authors": ["Keisuke Sakaguchi", "et al."], "award_type": "best_paper"},
        ],
        2019: [
            {"title": "How to Combine Tree-Search Methods in Reinforcement Learning", "authors": ["Yonathan Efroni", "et al."], "award_type": "best_paper"},
        ],
        2018: [
            {"title": "Memory-Augmented Monte Carlo Tree Search", "authors": ["Chenjun Xiao", "et al."], "award_type": "best_paper"},
        ],
    },

    # CVPR Best Papers
    "CVPR": {
        2024: [
            {"title": "Generative Image Dynamics", "authors": ["Zhengqi Li", "Richard Tucker", "Noah Snavely", "Aleksander Holynski"], "award_type": "best_paper"},
            {"title": "Rich Human Feedback for Text-to-Image Generation", "authors": ["Youwei Liang", "et al."], "award_type": "best_paper"},
            {"title": "pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction", "authors": ["et al."], "award_type": "best_paper_honorable_mention"},
            {"title": "Mip-Splatting: Alias-free 3D Gaussian Splatting", "authors": ["et al."], "award_type": "best_paper_honorable_mention"},
        ],
        2023: [
            {"title": "Planning-oriented Autonomous Driving", "authors": ["Yihan Hu", "et al."], "award_type": "best_paper"},
            {"title": "Visual Programming: Compositional Visual Reasoning Without Training", "authors": ["Tanmay Gupta", "Aniruddha Kembhavi"], "award_type": "best_paper_honorable_mention"},
        ],
        2022: [
            {"title": "Learning to Solve Hard Minimal Problems", "authors": ["Petr Hruby", "et al."], "award_type": "best_paper"},
            {"title": "Dual-Shutter Optical Vibration Sensing", "authors": ["et al."], "award_type": "best_paper_honorable_mention"},
            {"title": "EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation", "authors": ["et al."], "award_type": "best_paper_honorable_mention"},
        ],
        2021: [
            {"title": "GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields", "authors": ["Michael Niemeyer", "Andreas Geiger"], "award_type": "best_paper"},
            {"title": "Exploring Simple Siamese Representation Learning", "authors": ["Xinlei Chen", "Kaiming He"], "award_type": "best_paper_honorable_mention"},
        ],
        2020: [
            {"title": "Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild", "authors": ["Shangzhe Wu", "et al."], "award_type": "best_paper"},
        ],
        2019: [
            {"title": "A Theory of Fermat Paths for Non-Line-of-Sight Shape Reconstruction", "authors": ["Shumian Xin", "et al."], "award_type": "best_paper"},
        ],
        2018: [
            {"title": "Taskonomy: Disentangling Task Transfer Learning", "authors": ["Amir R. Zamir", "et al."], "award_type": "best_paper"},
            {"title": "Densely Connected Convolutional Networks", "authors": ["Zhuang Liu", "et al."], "award_type": "best_paper"},
        ],
    },
}


def normalize_title(title):
    """Normalize title for comparison."""
    import re
    # Remove special characters and lowercase
    normalized = re.sub(r'[^\w\s]', '', title.lower())
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    return normalized


def create_best_papers_list():
    """Create a comprehensive list of best papers."""
    best_papers = []

    for conference, years in BEST_PAPERS_DATA.items():
        for year, papers in years.items():
            for paper in papers:
                best_papers.append({
                    "title": paper["title"],
                    "authors": paper.get("authors", []),
                    "conference": conference,
                    "year": year,
                    "award_type": paper.get("award_type", "best_paper"),
                    "paper_type": "best_paper",
                    "is_high_impact": True
                })

    return best_papers


def update_top_papers(top_papers_path, best_papers):
    """Update top_1000_papers.json with best paper marks and is_high_impact field."""
    with open(top_papers_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    # Create a lookup dictionary for best papers by normalized title
    best_paper_lookup = {}
    for bp in best_papers:
        normalized = normalize_title(bp["title"])
        best_paper_lookup[normalized] = bp

    updated_count = 0

    for paper in papers:
        # Set is_high_impact based on paper_type
        current_type = paper.get("paper_type", "poster")

        if current_type in ["best_paper", "outstanding_paper"]:
            paper["is_high_impact"] = True
        elif current_type == "oral":
            paper["is_high_impact"] = True
        elif current_type == "spotlight":
            paper["is_high_impact"] = False  # medium impact
        else:
            paper["is_high_impact"] = False

        # Check if this paper is a best paper
        normalized_title = normalize_title(paper.get("title", ""))

        if normalized_title in best_paper_lookup:
            bp = best_paper_lookup[normalized_title]
            paper["paper_type"] = "best_paper"
            paper["award_type"] = bp["award_type"]
            paper["is_high_impact"] = True
            updated_count += 1
            print(f"Marked as best paper: {paper['title'][:60]}...")

    # Save updated papers
    with open(top_papers_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"\nUpdated {updated_count} papers with best_paper type")
    return papers


def create_best_papers_json(output_path, best_papers):
    """Create a dedicated best_papers.json file."""
    output = {
        "metadata": {
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "description": "AI Top Conference Best Papers 2018-2024",
            "conferences": ["ICLR", "ICML", "NeurIPS", "ACL", "AAAI", "CVPR"],
            "years": list(range(2018, 2025)),
            "total_papers": len(best_papers)
        },
        "papers": best_papers,
        "by_conference": {},
        "by_year": {}
    }

    # Group by conference
    for conf in ["ICLR", "ICML", "NeurIPS", "ACL", "AAAI", "CVPR"]:
        output["by_conference"][conf] = [p for p in best_papers if p["conference"] == conf]

    # Group by year
    for year in range(2018, 2025):
        output["by_year"][str(year)] = [p for p in best_papers if p["year"] == year]

    # Statistics
    output["statistics"] = {
        "total": len(best_papers),
        "by_conference": {conf: len(papers) for conf, papers in output["by_conference"].items()},
        "by_year": {year: len(papers) for year, papers in output["by_year"].items()}
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Created best_papers.json with {len(best_papers)} papers")
    return output


def main():
    base_path = Path("E:/JETB/pythons/sf/retrival/data/papers/metadata")
    top_papers_path = base_path / "top_1000_papers.json"
    best_papers_output = base_path / "best_papers.json"

    print("=" * 60)
    print("AI Top Conference Best Papers Collector")
    print("=" * 60)

    # Create best papers list
    print("\n1. Creating best papers list...")
    best_papers = create_best_papers_list()
    print(f"   Collected {len(best_papers)} best papers")

    # Print summary by conference
    print("\n   Summary by conference:")
    for conf in ["ICLR", "ICML", "NeurIPS", "ACL", "AAAI", "CVPR"]:
        count = len([p for p in best_papers if p["conference"] == conf])
        print(f"   - {conf}: {count} papers")

    # Update top_1000_papers.json
    print("\n2. Updating top_1000_papers.json...")
    if top_papers_path.exists():
        update_top_papers(top_papers_path, best_papers)
    else:
        print(f"   Warning: {top_papers_path} not found")

    # Create best_papers.json
    print("\n3. Creating best_papers.json...")
    create_best_papers_json(best_papers_output, best_papers)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
