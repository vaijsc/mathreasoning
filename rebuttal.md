Thank you for your reviews. We address your concerns as follows:

1. *“the drop in performance of current models when knowledge conflicts exist is striking. I am wondering if the way in which the answers are measured is not fully reflective of the correctness of the answers? See the points below for my elaboration.
A model that provides a consistent answer with the database does not necessarily mean that it is a good answer. The reason is specificity. For instance, if you ask a model "when was the US civil war" and it says "in the 19th century" vs. "1861", "1861-1865", or "April 12, 1861 - May 26, 1865", the first answer is least likely to conflict with answers in the database but it is also the least useful in the sense that it is very vague. Can the authors comment on this tradeoff between a model providing a correct but non-specific answer, and the model attempting to provide a specific answer? My concern is that measuring a model simply by correctness (lack of contradiction) is not sufficient.”*
    
    * Our process for collecting supporting contexts (lines 165-181) ensures that there are always direct references to the golden answers in the reference context, similar to extractive QA datasets (SQuaD, HotpotQA). Since the model is exposed to the expected answer, we anticipate that it will generate an answer with the same level of specificity.

2. *“Related to the above, oftentimes with RAG systems there may be multiple answers that would all be deemed the same by a user but not by a model, simply due to subtle changes in formatting, e.g., "April 15, 1990", "15 April 1990", "04/15/1990", "15-04-1990", etc. Are the accuracy computations computed based on literal string matching, or is there a more robust method to interpret if answers "mean" the same thing”*

    * From lines 159-160 in our submission, what we meant by "Wikipedia recommended answer variations" may address your concern. For datetime strings, we refer to the Mediawiki Wikibase Data Model (Dates and time section) to interpret how datetime values are represented in Wikidata. We then implement a function to translate a datetime value stored in Wikidata to all of its suggested written formats as advised by the Wikipedia Manual of Style. Except for questions asking for datetime values, all other questions in our dataset require answers without numerical values (see Table 5 in the Appendix), which are already covered by Wikidata contributors. For example, in Wikidata, the author "J. R. R. Tolkien" can be referred to by various names, such as:
        * J-R-R Tolkien
        * Tolkien
        * John Ronald Reuel Tolkien
        * John Tolkien
        * J.R.R Tolkien
        * J.R.R. Tolkien
        * John R. R. Tolkien
        We will incorporate the information above into the revised version of our paper.

3. *“To what degree should a model infer what the user likely meant? For instance, if I ask "When was Adam Smith born?", most users probably mean the famous economist even though there are likely thousands of Adam Smith's that have existed.”*
    
    * As stated in our submission, from lines 8-11 and 54-58, we expect the model to be faithful to the retrieved contexts. Within the scope of this work, to make our evaluation simple and automatic, we only consider the accuracy of the answers. In this case, we count an LLM's answer as correct if it correctly states all the different birth dates of Adam Smith mentioned in the retrieved contexts, ensuring no information is missed. In practice, the number of retrieved contexts in an LLM input is typically not too large (5-10 contexts). 

4. *“How pedantic should models be when evaluating answers?”*
    
    * For future works, we suggest that more needs to be done to evaluate the quality of model answers in terms of clarity and faithfulness. Clarity: Models should distinguish between different answers. For example, if there are three different Adam Smiths, with two sharing the same birthdate, LLMs should clarify that there are three different individuals, in addition to providing the birthdates. Faithfulness: To support the user's decision, citations should be provided, along with a proper count of the number of supporting paragraphs for each answer. However, this is out of the scope of this work. 

5. *“Ethical Concerns: There is a certain degree of guessing what the user meant in any RAG system. This guessing process can contain biases. It may be worth stating this.”*
    
    * In WhoQA, we want to highlight that knowledge conflicts can occur even if all retrieved contexts are truthful. In such situations, LLMs should mention all available and applicable information in their answers. From this perspective, the responsibility of ensuring truthfulness and fairness in context retrieval falls on the retrieval module. While efforts like Page Rank aim to ensure the truthfulness and credibility of query results, fairness in the context retrieval module can be an area of future research. We will incorporate this discussion into the revised version of our paper.


Thank you for your reviews. We address your concerns as follows:

1. *“1. There is a related subtopic[A] on asking clarifying questions in presence of conflicts, especially in chat settings. it would have nice to have discussions/experiments related to that. [A] : https://arxiv.org/abs/2305.15933”*
    
    * In this work, we consider a general setting (discussed from lines 85 to 90) where sometimes there is no information to resolve ambiguity. For example, the statements "George Washington is a farmer" and "George Washington is the first president of the United States" both refer to the same person but can seem contradictory if found in two separate documents. If there is no further information from the documents, it is difficult to determine whether the two mentions of George Washington refer to one person or two different individuals.

2. *"Sometimes more verbose prompt/instruction on how to deal with conflict works well and any such exploration on finding the right prompt template (especially for W/oS and W/S) is missing. Alluding to the previous point (weaknesses pt 2), one would safely assume that a well instructed prompt for W/S will improve the performance form W/oS but that is not the case for Llama3. Some more analysis comparing the response from llama3 w/s and w/os would have improved the quality of the contributions.”*

    * As shown in our results in Table 1, only the 8B version of LLama 3 has its performance in the W/S setting slightly lower than in the W/oS setting (71.5% compared to 72.4%, respectively). We also manually reviewed LLama 3's answers and discussed our findings in Section 3.4 (lines 298 to 305). To some extent, models from the LLama 3 series already have an inherent ability to detect knowledge conflict, regardless of whether the presence of conflicts is stated. However, this does not mean that LLama 3 has completely solved the problem of knowledge conflict, as there is still a performance gap to address.



Thank you for your reviews. We address your concerns as follows:

1. *“The reason to construct 5 question templates per property is not clear to me. Could you further explain? (line 150)”*
    
    * For each entity and its selected properties, our 5-question template is used to generate 5 different questions with roughly the same semantics. To elaborate on our description from lines 170-174, we average our candidate contexts' relevance scores over the 5 different generated questions to improve the robustness of selecting relevant contexts. We will incorporate this information into the revised version of our paper. 

2. *“The evaluation metric may be too strict for WhoQA compared to SimQA. The model only needs to answer one answer correctly in SimQA while multiple in WhoQA. The model performance degradation may also be attributed to the complexity of the question+context itself. A fairer comparison could be: ask LLMs several independent simple questions where the context does not involve knowledge conflicts, and compare the performance with WhoQA.”*
    
    * As noted in the caption of Table 1, “SimQA” denotes our Simple QA setting (Section 3.2, line 235), in which we provide LLMs with every single context from WhoQA, ensuring that no conflicting information is presented in the input context. The results shown in our Simple QA setting highlight the ease of finding answers within the individual contexts of WhoQA, allowing our subsequent experiments (Sections 3.3 and 3.4) to focus more on the issue of knowledge conflicts. 
                                                                                                                   
3. *“Could you provide details of human annotation?”*
    * We would like to add more details to our annotation process (lines 193-207). For each sample, we provide an annotator with a question, a set of answers from Wikidata, and a supporting context. We highlight every appearance of all answers in the supporting context to help annotators find the answers more quickly. An annotator marks a sample as correct if all the answers in the answer set can be inferred from the supporting context and match the given question. Otherwise, the sample is excluded. We only keep samples that all annotators mark as correct. We will incorporate the information above into the revised version of our paper.

4. *“In line 231-234, have you performed any error analysis to validate this assumption?”*
    
    * An entity (human, place, etc.) can have different references that appear in various sources. For example, in Wikidata, the author “J. R. R. Tolkien” has the following available names that can be referred to in different sources:
        - J-R-R Tolkien
        - Tolkien
        - John Ronald Reuel Tolkien
        - John Tolkien
        - J.R.R Tolkien
        - J.R.R. Tolkien
        - John R. R. Tolkien
    
        Collecting an exhaustive set of all possible references for each entity is impossible; therefore, we rely on the information provided by Wikidata, which is already comprehensive. Please refer to the Wikidata Help page about Aliases for the criteria for references for each entity. Moreover, choosing Wikidata/Wikipedia as the source of answers is also applied in other works [1][2].
        
        [1] : When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories. ACL 2023.
        
        [2]: AMBIGQA: Answering Ambiguous Open-domain Questions. EMNLP 2020.


5. *“It might also be interesting to experiment when no context is provided and analyze the alignment with the model performance when conflicted supporting context are provided.”*
    
    * Our work specifically addresses scenarios where conflicts arise from supporting contexts. If no context is provided, such situations fall outside the scope of our study. This focus ensures a thorough examination of how conflicting information within contexts impacts the performance of models, highlighting the critical importance of context in understanding and resolving ambiguities.

6. *“Datasets: 1 = No usable datasets submitted.”*
    
    * We mentioned in the Introduction section that upon acceptance, we will release the dataset. Currently, during the review process, we are using the placeholder "anonymous" URL to comply with the anonymous submission policy.




Thanks for your time and consideration. We would like to address your remaining concerns as follows:

>“For 3., could you further explain how the annotators are selected and what is the iaa in revised version?” 

* Our hired annotators are two graduated bachelor, with scientific background. We let them do two IELTS reading tests and none of them receive any test result bellow 8.0. 

    As each annotator has to annotate whether or not the provided answers in Wikidata can be inferred from a context, the following table denotes the numbers of annotated contexts in the two labels by the two annotators:

|                             	| **Correct (Annotator 1)** 	| **Incorrect (Annotator 1)** 	|
|-----------------------------	|:---------------------------:	|:-----------------------------:|
| **Correct (Annotator 2)**   	|           17365           	|              77             	|
| **Incorrect (Annotator 2)** 	|             59            	|             1466            	|

* From the table, a Cohen's Kappa score of 0.95 can be calculated. As mentioned in line 200, we exclude 1602 contexts, including those which the two annotators have differrent answers. We will incorporate the information above into the revised version of our paper. 

>“For 5., I was expecting to apply error analysis to a small subset of the responses and check what is the percentage of error from missing correct answers so as to make your argument in lines 231-234 more convincing.”

* Thank you for your insightful suggestion. As the discussion deadline is comming close, we may not have enough time to include such an analysis. We will do that in the next revision of our paper. 
