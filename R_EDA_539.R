library(tidyverse)
library(reshape2) #for the melt function

cvrp <- read.csv(file="CVRP_dataset.csv")

################### Feature Distributions ################

# Features 1 - 4
d <- melt(cvrp[c(7:10)])
ggplot(data = d, mapping = aes(x = value)) + 
  facet_wrap(~variable, scales = 'free') +
  geom_histogram() + 
  labs(x = 'Feature Value', y = 'Count') + #, title = 'Histograms of Features 1 - 4') +
  theme_classic()+
  theme(plot.title = element_text(hjust = 0.5))

# Features 5 - 8
d <- melt(cvrp[c(11:14)])
ggplot(data = d, mapping = aes(x = value)) + 
  facet_wrap(~variable, scales = 'free') +
  geom_histogram() + 
  labs(x = 'Feature Value', y = 'Count') + #, title = 'Histograms of Features 5 - 8') +
  theme_classic()+
  theme(plot.title = element_text(hjust = 0.5))

# Features 9 - 12
d <- melt(cvrp[c(15:18)])
ggplot(data = d, mapping = aes(x = value)) + 
  facet_wrap(~variable, scales = 'free') +
  geom_histogram() + 
  labs(x = 'Feature Value', y = 'Count') + #, title = 'Histograms of Features 9 - 12') +
  theme_classic()+
  theme(plot.title = element_text(hjust = 0.5))

# Features 13 - 16
d <- melt(cvrp[c(19:22)])
ggplot(data = d, mapping = aes(x = value)) + 
  facet_wrap(~variable, scales = 'free') +
  geom_histogram() + 
  labs(x = 'Feature Value', y = 'Count') + #, title = 'Histograms of Features 13 - 16') +
  theme_classic()+
  theme(plot.title = element_text(hjust = 0.5))

# Features 17 - 20
d <- melt(cvrp[c(23:26)])
ggplot(data = d, mapping = aes(x = value)) + 
  facet_wrap(~variable, scales = 'free') +
  geom_histogram() + 
  labs(x = 'Feature Value', y = 'Count') + #, title = 'Histograms of Features 17 - 20') +
  theme_classic()+
  theme(plot.title = element_text(hjust = 0.5))

# Features 21 - 23
d <- melt(cvrp[c(27:29)])
ggplot(data = d, mapping = aes(x = value)) + 
  facet_wrap(~variable, scales = 'free', nrow = 2) +
  geom_histogram() + 
  labs(x = 'Feature Value', y = 'Count') + #, title = 'Histograms of Features 21 - 23') +
  theme_classic()+
  theme(plot.title = element_text(hjust = 0.5))

############ Label Distribution #####################
# put in descending order

# summary table of labels
counts = cvrp %>%
  group_by(Label) %>%
  summarize(counts = length(Label))

# get Label in descending order based upon its count value
counts$Label = factor(counts$Label,                                   
                      levels = counts$Label[order(counts$counts, decreasing = TRUE)])

# create bar chart
ggplot(data = counts, mapping = aes(x = Label, y = counts, fill = Label)) + 
  geom_bar(stat = 'identity')+
  labs(x = 'Label', y = 'Count', title = 'Algorithm Labels in CVRP Data Set')+
  geom_text(aes(label = counts), vjust = 1.5, color = 'white', fontface = 'bold')+
  theme_classic() + 
  theme(plot.title = element_text(size=16))+
  theme(legend.position = 'none')

###### Create grouped boxplots of Labels in CVRP data ###################
# avoid the copy/paste of above

#list of vectors for feature columns to make
features_list = list(c(7:10), c(11:14), c(15:18), c(19:22), c(23:26), c(27:29))

for (features in features_list) {
  
  # put cvrp data into long format for ggplot
  cvrp_long = melt(data = cvrp, id.vars = c(features, 34), measure.vars = features,
                   variable.name = 'Feature', value.name = 'Value')
  
  # put Label in same order as the bar chart
  cvrp_long$Label = factor(cvrp_long$Label,                                   
                           levels = c('CW', 'SOM', 'GA', 'SP'))
  
  # generate/print the plots for 
  print(
    ggplot(data = cvrp_long, mapping = aes(x = Label, y = Value, color = Label)) + 
      facet_wrap(~Feature, scales = 'free', nrow = 2) +
      geom_boxplot() + 
      labs(x = 'Algorithm Label', y = 'Feature Value') + #, title = 'Histograms of Features 21 - 23') +
      theme_classic()+
      theme(plot.title = element_text(hjust = 0.5))+
      theme(legend.position = 'none')
  )
}




