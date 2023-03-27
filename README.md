# MasterThesis
This is a repository for the graduation project for the Master thesis in Data Science for Life Sciences at Hanze University of Applied Sciences.

Arbuscular mycorrhizal fungi (AMF) are symbiotic fungi that form complex networks underground to exchange nutrients with carbon resources from their host plants. The structure of the networks will play an important role in trading dynamics. The fungus may decide to invest in a different structure responsible for absorbing phosphorous (branched absorbing structure (BAS)) or exploration (runner hyphae (RH)). The fungus will also invest in different widths of hyphae to optimize its resource acquisition strategy. In this thesis, the evolution of hyphal width was followed over time and space to unravel the strategy of the fungus in terms of width allocation. A convolutional neural network (CNN) model with a mean absolute error of 0.759 µm was developed to track the width evolution from a high-resolution automated imaging setup. The one-dimension vector from image data was extracted as a feature and the actual width was computed by pixel from high magnification image as a label for training the model. AMF can double its width in 40 hours and stay at a stable phase after 58 hours. The vasculature network formed by AMF does not follow Murray’s law. There is a threshold to distinguish between the RH to BAS in the junction.

Student: Kai-Kai Lin ka.lin@st.hanze.nl  
Supervisor: Tsjerk Wassenaar t.a.wassenaar@pl.hanze.nl  
Daily supervisor: Coretin Bisot C.Bisot@amolf.nl  

# Research questions
•	How does the width evolve between different hyphae segments over time?  
•	Is there a relationship between the width of the different hyphae edges at the intersection?  
•	What criteria can be used to categorize RH and BAS?  

# Requirements
