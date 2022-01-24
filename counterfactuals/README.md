# Counterfactual Explanation (Actionable Recourse)

Counterfactual explanations mainly target to find the mimimum perturbation which changes the original prediction(Ususlly from an undesirable prediction to ideal one). The perturbation itself is a valid instance following the real data distribution as the training samples. It has broad applications, E.g., finance, education, health care ect. Specifically, what should I do to get the credit card approved if I received the rejection. This task can be viewed as extracting knowledge/solutions from the black-box models. It belongs to the instance-level explanation. Quite interesting to dive deeper!!!

The two use cases of counterfactual explanations:

![counterfactual explanations](https://github.com/iversonicter/awesome-explainable-ai/blob/master/fig/cf.png)

## Survey papers 

[Counterfactual Explanations for Functional Data:A Mathematical Optimization Approach](https://www.researchgate.net/profile/Jasone-Ramirez-Ayerbe/publication/357700882_Counterfactual_Explanations_for_Functional_Data_A_Mathematical_Optimization_Approach/links/61dbed08e669ee0f5c997e79/Counterfactual-Explanations-for-Functional-Data-A-Mathematical-Optimization-Approach.pdf), Preprint 2021

[Counterfactuals and Causability in Explainable Artificial Intelligence: Theory, Algorithms, and Applications](https://arxiv.org/abs/2103.04244), Arxiv preprint 2021

[A Survey of Contrastive and Counterfactual Explanation Generation Methods for Explainable Artificial Intelligence](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9321372), IEEE Access 2021

[Good Counterfactuals and Where to Find Them: A Case-Based Technique for Generating Counterfactuals for Explainable AI (XAI)](https://arxiv.org/pdf/2005.13997.pdf), Arxiv preprint 2020

[Counterfactual Explanations for Machine Learning: A Review](https://arxiv.org/pdf/2010.10596.pdf), Arxiv preprint 2020

[A survey of algorithmic recourse: definitions, formulations, solutions, and prospects](https://arxiv.org/abs/2010.04050), Arxiv preprint 2020

[On the computation of counterfactual explanations -- A survey](https://arxiv.org/abs/1911.07749), Arxiv preprint 2019

[Issues with post-hoc counterfactual explanations: a discussion](https://arxiv.org/abs/1906.04774), ICML Workshop 2019

[Counterfactuals in Explainable Artificial Intelligence (XAI): Evidence from Human Reasoning](https://www.ijcai.org/Proceedings/2019/0876.pdf), IJCAI 2019

[CONTRASTIVE EXPLANATION: A STRUCTURAL-MODEL APPROACH](https://arxiv.org/pdf/1811.03163.pdf), Arxiv preprint 2018

## Papers
[On the Fairness of Causal Algorithmic Recourse](https://arxiv.org/pdf/2010.06529.pdf), AAAI 2022

[Unsupervised Editing for Counterfactual Stories](https://arxiv.org/pdf/2112.05417.pdf), AAAI 2022

[Counterfactual explanation of Bayesian model uncertainty](https://link.springer.com/article/10.1007/s00521-021-06528-z), Nerual Computing and Applications 2021

[Learning Models for Actionable Recourse](https://arxiv.org/abs/2011.06146), NeurIPS 2021

[Counterfactual Explanations in Sequential Decision Making Under Uncertainty](https://arxiv.org/abs/2107.02776), NeurIPS 2021

[Counterfactual Explanations Can Be Manipulated](https://arxiv.org/abs/2106.02666), NeurIPS 2021

[The Skyline of Counterfactual Explanations for Machine Learning Decision Models](https://dl.acm.org/doi/abs/10.1145/3459637.3482397), CIKM 2021

[Counterfactual Explanations for Optimization-Based Decisions in the Context of the GDPR](https://www.ijcai.org/proceedings/2021/0564.pdf), IJCAI 2021

[Robust Counterfactual Explanations on Graph Neural Network](https://arxiv.org/pdf/2107.04086.pdf), Arxiv 2021

[CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks](https://arxiv.org/abs/2102.03322), DLG-KDD 2021, [code](https://github.com/a-lucic/cf-gnnexplainer)

[A Language for Counterfactual Generative Models](http://proceedings.mlr.press/v139/tavares21a/tavares21a.pdf), ICML 2021

[Optimal Counterfactual Explanations in Tree Ensembles](https://arxiv.org/abs/2106.06631), ICML 2021

[Model-Based Counterfactual Synthesizer for Interpretation](https://arxiv.org/pdf/2106.08971.pdf), KDD 2021

[Leveraging Latent Features for Local Explanations](https://arxiv.org/pdf/1905.12698.pdf), KDD 2021

[Counterfactual Graphs for Explainable Classification of Brain Networks](https://arxiv.org/abs/2106.08640), KDD 2021

[A Few Good Counterfactuals: Generating Interpretable, Plausible & Diverse Counterfactual Explanations](https://arxiv.org/pdf/2101.09056.pdf), Arxiv preprint 2021

[Model-agnostic and Scalable Counterfactual Explanations via Reinforcement Learning](https://arxiv.org/pdf/2106.02597.pdf), Arxiv preprint 2021

[GeCo: Quality Counterfactual Explanations in Real Time](https://arxiv.org/pdf/2101.01292.pdf), Arxiv preprint 2021

[Generating Interpretable Counterfactual Explanations By Implicit Minimisation of Epistemic and Aleatoric Uncertainties](https://arxiv.org/abs/2103.08951), AISTATS 2021, [code](https://github.com/oscarkey/explanations-by-minimizing-uncertainty)

[FIMAP: Feature Importance by Minimal Adversarial Perturbation](https://www.aaai.org/AAAI21Papers/AAAI-2721.Chapman-RoundsM.pdf), AAAI 2021

[Generate Your Counterfactuals: Towards Controlled Counterfactual Generation for Text](https://arxiv.org/pdf/2012.04698.pdf), AAAI 2021

[Counterfactual Fairness with Disentangled Causal Effect Variational Autoencoder](https://arxiv.org/pdf/2011.11878.pdf), AAAI 2021

[Counterfactual Explanations for Oblique Decision Trees: Exact, Efficient Algorithms](https://arxiv.org/abs/2103.01096), AAAI 2021

[Ordered Counterfactual Explanation by Mixed-Integer Linear Optimization](https://arxiv.org/abs/2012.11782), AAAI 2021

[Interpretability Through Invertibility: A Deep Convolutional Network With Ideal Counterfactuals And Isosurfaces](https://openreview.net/forum?id=8YFhXYe1Ps)

[Counterfactual Generative Networks](http://www.cvlibs.net/publications/Sauer2021ICLR.pdf), ICLR 2021, [code](https://github.com/autonomousvision/counterfactual_generative_networks)

[Learning "What-if" Explanations for Sequential Decision-Making](https://openreview.net/forum?id=h0de3QWtGG), ICLR 2021

[GRACE: Generating Concise and Informative Contrastive Sample to Explain Neural Network Model’s Prediction](https://arxiv.org/abs/1911.02042), KDD 2020 [code](https://github.com/lethaiq/GRACE_KDD20)

[An ASP-Based Approach to Counterfactual Explanations for Classification](https://arxiv.org/pdf/2004.13237.pdf), RuleML + PR 2020

[DECE: Decision Explorer with Counterfactual Explanations for Machine Learning Models](https://arxiv.org/pdf/2008.08353.pdf), TVCG 2020

[Good Counterfactuals and Where to Find Them: A Case-Based Technique for Generating Counterfactuals for Explainable AI (XAI)](https://arxiv.org/pdf/2005.13997.pdf), ICCBR 2020

[Learning Global Transparent Models from Local Contrastive Explanations](https://arxiv.org/abs/2002.08247), NeurIPS 2020

[Decisions, Counterfactual Explanations and Strategic Behavior](https://arxiv.org/abs/2002.04333), NeurIPS 2020

[Algorithmic recourse under imperfect causal knowledge a probabilistic approach](https://arxiv.org/abs/2006.06831), NeurIPS 2020

[Beyond Individualized Recourse: Interpretable and Interactive Summaries of Actionable Recourses](https://arxiv.org/abs/2009.07165), NeurIPS 2020

[Learning What Makes a Difference from Counterfactual Examples and Gradient Supervision](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550579.pdf), ECCV 2020

[Counterfactual Vision-and-Language Navigation via Adversarial Path Sampler](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510069.pdf), ECCV 2020

[CERTIFAI: Counterfactual Explanations for Robustness, Transparency, Interpretability, and Fairness of Artificial Intelligence models](https://arxiv.org/pdf/1905.07857.pdf),  AAAI/AIES 2020

[FACE: Feasible and Actionable Counterfactual Explanation](https://dl.acm.org/doi/abs/10.1145/3375627.3375850), AAAI/AIES 2020

[Counterfactual Explanations & Adversarial Examples](https://arxiv.org/pdf/2009.05487.pdf), Arxiv preprint 2020

[Instance-Based Counterfactual Explanations for Time Series Classification](https://arxiv.org/pdf/2009.13211.pdf), Arxiv preprint 2020

[On Relating 'Why?' and 'Why Not?' Explanations](https://arxiv.org/abs/2012.11067), Arxiv preprint 2020

[Model extraction from counterfactual explanations](https://arxiv.org/pdf/2009.01884.pdf), Arxiv preprint 2020

[Efficient computation of counterfactual explanations of LVQ models](https://arxiv.org/pdf/1908.00735.pdf), Arxiv preprint 2020

[Plausible Counterfactuals: Auditing Deep Learning Classifiers with Realistic Adversarial Examples](https://arxiv.org/pdf/2003.11323.pdf), Arxiv preprint 2020

[ViCE: Visual Counterfactual Explanations for Machine Learning Models](https://arxiv.org/pdf/2003.02428.pdf), IUI 2020

[Model extraction from counterfactual explanations](https://arxiv.org/pdf/2009.01884.pdf), Arxiv 2020

[Scaling Guarantees for Nearest Counterfactual Explanations](https://arxiv.org/abs/2010.04965), Arxiv preprint 2020

[PermuteAttack: Counterfactual Explanation of Machine Learning Credit Scorecards](https://arxiv.org/pdf/2008.10138.pdf), Arxiv preprint 2020

[Decisions, Counterfactual Explanations and Strategic Behavior](https://arxiv.org/abs/2002.04333), Arxiv preprint 2020

[FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles](https://arxiv.org/abs/1911.12199), Arxiv preprint 2020

[Interpretable and Interactive Summaries of Actionable Recourses](https://arxiv.org/abs/2009.07165), Arxiv preprint 2020

[Counterfactual Explanation Based on Gradual Construction for Deep Networks](https://arxiv.org/abs/2008.01897), Arxiv preprint 2020

[CounteRGAN: Generating Realistic Counterfactuals with Residual Generative Adversarial Nets](https://arxiv.org/pdf/2009.05199.pdf), Arxiv preprint 2020

[EXPLAINABLE IMAGE CLASSIFICATION WITH EVIDENCE COUNTERFACTUAL](https://arxiv.org/pdf/2004.07511.pdf), Arxiv preprint 2020

[Model-Agnostic Counterfactual Explanations for Consequential Decisions](https://arxiv.org/abs/1905.11190), AISTATS 2020

[Explaining Data-Driven Decisions made by AI Systems: The Counterfactual Approach](https://arxiv.org/pdf/2001.07417.pdf), Arxiv preprint 2020

[On the Fairness of Causal Algorithmic Recourse](https://arxiv.org/pdf/2010.06529.pdf), Arxiv preprint 2020

[On Counterfactual Explanations under Predictive Multiplicity](http://proceedings.mlr.press/v124/pawelczyk20a.html), UAI 2020

[EXPLANATION BY PROGRESSIVE EXAGGERATION](https://iclr.cc/virtual_2020/poster_H1xFWgrFPS.html), ICLR 2020

[Algorithmic Recourse: from Counterfactual Explanations to Interventions](https://arxiv.org/abs/2002.06278), Arxiv preprint 2020

[Learning Model-Agnostic Counterfactual Explanations for Tabular Data](https://dl.acm.org/doi/abs/10.1145/3366423.3380087),  ACM WWW 2020, [code](https://github.com/MartinPawel/c-chvae)

[The hidden assumptions behind counterfactual explanations and principal reasons](https://arxiv.org/pdf/1912.04930.pdf), ACM Facct 2020

[Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations](https://arxiv.org/pdf/1905.07697.pdf), ACM Facct 2020

[The philosophical basis of algorithmic recourse](https://dl.acm.org/doi/abs/10.1145/3351095.3372876), ACM Facct 2020

[Why Does My Model Fail? Contrastive Local Explanations for Retail Forecasting](https://arxiv.org/pdf/1908.00085.pdf), ACM Facct 2020

[Convex Density Constraints for Computing Plausible Counterfactual Explanations](https://arxiv.org/pdf/2002.04862.pdf), ICANN 2020

[Fast Real-time Counterfactual Explanations](https://arxiv.org/pdf/2007.05684.pdf), ICML 2020 Workshop

[CRUDS: Counterfactual Recourse Using Disentangled Subspaces](https://finale.seas.harvard.edu/files/finale/files/cruds-_counterfactual_recourse_using_disentangled_subspaces.pdf), ICML 2020 Workshop

[DACE: Distribution-Aware Counterfactual Explanation by Mixed-Integer Linear Optimization](https://www.ijcai.org/Proceedings/2020/0395.pdf), IJCAI 2020

[Relation-Based Counterfactual Explanations for Bayesian Network Classifiers](https://www.ijcai.org/Proceedings/2020/63), IJCAI 2020

[PRINCE: Provider-side Interpretability with Counterfactual Explanations in Recommender Systems](https://arxiv.org/pdf/1911.08378.pdf), WSDM 2020

[CoCoX: Generating Conceptual and Counterfactual Explanations via Fault-Lines](https://aaai.org/ojs/index.php/AAAI/article/view/5643/5499), AAAI 2020

[SCOUT: Self-aware Discriminant Counterfactual Explanations](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_SCOUT_Self-Aware_Discriminant_Counterfactual_Explanations_CVPR_2020_paper.pdf), CVPR 2020, [code](https://github.com/peiwang062/SCOUT)

[Multi-Objective Counterfactual Explanations](https://arxiv.org/pdf/2004.11165.pdf), PPSN 2020

[EMAP: Explanation by Minimal Adversarial Perturbation](https://arxiv.org/pdf/1912.00872.pdf), AAAI 2020

[Random forest explainability using counterfactual sets](https://doi.org/10.1016/j.inffus.2020.07.001), Information Fusion 2020

[The Dangers of Post-hoc Interpretability: Unjustified Counterfactual Explanations](https://www.ijcai.org/proceedings/2019/388), IJCAI 2019

[Multimodal Explanations by Predicting Counterfactuality in Videos](https://arxiv.org/pdf/1812.01263.pdf), CVPR 2019

[Unjustified Classification Regions and Counterfactual Explanations In Machine Learning](https://ecmlpkdd2019.org/downloads/paper/226.pdf), ECML-PDKK 2019

[Factual and Counterfactual Explanations for Black Box Decision Making](https://ieeexplore.ieee.org/document/8920138), IEEE Intelligent Systems 2019

[Model Agnostic Contrastive Explanations for Structured Data](https://arxiv.org/pdf/1906.00117.pdf), Arxiv preprint 2019

[Counterfactuals uncover the modular structure of deep generative models](https://arxiv.org/pdf/1812.03253.pdf), Arxiv preprint 2019

[Actionable Interpretability through Optimizable Counterfactual Explanations for Tree Ensembles](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/lucic-2019-actionable-arxiv.pdf), arxiv preprint 2019

[Towards Realistic Individual Recourse and Actionable Explanations in Black-Box Decision Making Systems](https://arxiv.org/pdf/1907.09615.pdf), Arxiv preprint 2019

[Generative Counterfactual Introspection for Explainable Deep Learning](https://arxiv.org/abs/1907.03077), Arxiv preprint 2019

[Generating Counterfactual and Contrastive Explanations using SHAP](https://arxiv.org/pdf/1906.09293.pdf), Arxiv preprint 2019

[Interpretable Counterfactual Explanations Guided by Prototypes](https://arxiv.org/pdf/1907.02584.pdf), Arxiv preprint 2019

[Counterfactual Visual Explanations](https://arxiv.org/pdf/1904.07451.pdf), ICML 2019

[Counterfactual Explanation Algorithms for Behavioral and Textual Data](https://arxiv.org/abs/1912.01819), Arxiv preprint 2019

[Synthesizing Action Sequences for Modifying Model Decisions](https://arxiv.org/pdf/1910.00057.pdf), Arxiv preprint 2019

[Measurable Counterfactual Local Explanations for Any Classifier](https://arxiv.org/pdf/1908.03020.pdf), Arxiv preprint 2019

[Generating Contrastive Explanations with Monotonic Attribute Functions](https://arxiv.org/pdf/1905.12698.pdf), arxiv preprint 2019

[The What-If Tool: Interactive Probing of Machine Learning Models](https://arxiv.org/pdf/1907.04135v2.pdf), TVCG 2019

[Preserving Causal Constraints in Counterfactual Explanations for Machine Learning Classifiers](https://arxiv.org/abs/1912.03277), NeurIPS Workshop 2019

[Actionable Recourse in Linear Classification](https://dl.acm.org/doi/pdf/10.1145/3287560.3287566), ACM Facct 2019

[EQUALIZING RECOURSE ACROSS GROUPS](https://arxiv.org/abs/1909.03166),  Arxiv preprint 2019

[Efficient Search for Diverse Coherent Explanations](https://arxiv.org/pdf/1901.04909.pdf), ACM Facct 2019

[Counterfactual explanations of machine learning predictions: opportunities and challenges for AI safety](http://ceur-ws.org/Vol-2301/paper_20.pdf), SafeAI@AAAI 2019

[Explaining image classifiers by counterfactual generation](https://arxiv.org/pdf/1807.08024.pdf), ICLR 2019

[Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR](https://arxiv.org/abs/1711.00399), Hardvard Journal of Law & Technology 2018 (strong recommend)

[Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives](https://papers.nips.cc/paper/7340-explanations-based-on-the-missing-towards-contrastive-explanations-with-pertinent-negatives), NIPS 2018, [code](https://github.com/IBM/Contrastive-Explanation-Method)

[Interpreting Neural Network Judgments via Minimal, Stable, and Symbolic Corrections](https://arxiv.org/pdf/1802.07384.pdf), NIPS 2018

[Interpretable Credit Application Predictions With Counterfactual Explanations](https://arxiv.org/abs/1811.05245), Arxiv preprint 2018

[Comparison-based Inverse Classification for Interpretability in Machine Learning](https://hal.sorbonne-universite.fr/hal-01905982/document), IPMU 2018

[Generating Counterfactual Explanations with Natural Language](https://arxiv.org/pdf/1806.09809.pdf), ICML 2018 Workshop

[Grounding Visual Explanations](https://arxiv.org/pdf/1807.09685.pdf), ECCV 2018

[Interpreting Neural Network Judgments via Minimal, Stable, and Symbolic Corrections](https://proceedings.neurips.cc/paper/2018/file/300891a62162b960cf02ce3827bb363c-Paper.pdf), NeurIPS 2018

[Inverse Classification for Comparison-based Interpretability in Machine Learning](https://arxiv.org/pdf/1712.08443.pdf), Arxiv 2017

[Interpretable Predictions of Tree-based Ensembles via Actionable Feature Tweaking](https://arxiv.org/pdf/1706.06691.pdf), KDD 2017

[When Worlds Collide: Integrating Different Counterfactual Assumptions in Fairness](https://proceedings.neurips.cc/paper/2017/file/1271a7029c9df08643b631b02cf9e116-Paper.pdf), NeurIPS 2017

[A budget-constrained inverse classification framework for smooth classifiers](https://arxiv.org/abs/1605.09068), ICDMW 2017

[Generalized Inverse Classification](https://epubs.siam.org/doi/pdf/10.1137/1.9781611974973.19), SIAM 2017

[Explaining Data-Driven Document Classifications](http://pages.stern.nyu.edu/~fprovost/Papers/MartensProvost_Explaining.pdf), MIS Quarterly 2014

[The Inverse Classification Problem](https://link.springer.com/article/10.1007/s11390-010-9337-x), Journal of Computer Science and Technology 2010

[The cost-minimizing inverse classification problem: a genetic algorithm approach](https://www.sciencedirect.com.remotexs.ntu.edu.sg/science/article/pii/S0167923600000774), Decision Support Systems 2000

## Github Repos

Alibi: [https://github.com/SeldonIO/alibi](https://github.com/SeldonIO/alibi) ![](https://img.shields.io/github/stars/SeldonIO/alibi.svg?style=social)

actionable-recourse: [https://github.com/ustunb/actionable-recourse](https://github.com/ustunb/actionable-recourse), Scikit-Learn ![](https://img.shields.io/github/stars/ustunb/actionable-recourse?style=social)

CEML: [https://github.com/andreArtelt/ceml](https://github.com/andreArtelt/ceml), Pytorch, Keras, Tensorflow, Scikit-Learn ![](https://img.shields.io/github/stars/andreArtelt/ceml?style=social)

Dice: [https://github.com/interpretml/DiCE.git](https://github.com/interpretml/DiCE.git), Pytorch, TensorFlow ![](https://img.shields.io/github/stars/interpretml/DiCE?style=social)

ContrastiveExplanation: [https://github.com/MarcelRobeer/ContrastiveExplanation](https://github.com/MarcelRobeer/ContrastiveExplanation), scikit-learn, ![](https://img.shields.io/github/stars/MarcelRobeer/ContrastiveExplanation?style=social)

cf-feasibility: [https://github.com/divyat09/cf-feasibility](https://github.com/divyat09/cf-feasibility), Pytorch, Tensorflow, Scikit-Learn, ![](https://img.shields.io/github/stars/divyat09/cf-feasibility?style=social)

Mace: [https://github.com/amirhk/mace](https://github.com/amirhk/mace), Scikit-Learn, ![](https://img.shields.io/github/stars/amirhk/mace?style=social)

Strategic-Decisions: [https://github.com/Networks-Learning/strategic-decisions](https://github.com/Networks-Learning/strategic-decisions), Scikit-learn ![](https://img.shields.io/github/stars/Networks-Learning/strategic-decisions?style=social)

Contrastive-Explanation-Method: [https://github.com/IBM/Contrastive-Explanation-Method](https://github.com/IBM/Contrastive-Explanation-Method), Tensorflow ![](https://img.shields.io/github/stars/IBM/Contrastive-Explanation-Method?style=social)
