# mclements20@georgefox.edu
# CSIS 485
# Final Project

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.model_selection import StratifiedKFold
import sklearn.svm


from sklearn.model_selection import train_test_split

pokemon = pd.read_csv("pokedex_(Update_05.20).csv", header=0, usecols=(0, 2, 4, 5, 6,  9, 10, 17, 18, 19, 20, 21,
                                                                       22, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                                                                       42, 43, 44, 45, 46, 47, 48, 49, 50),
                      names=("#", "name", "japanName", "gen", "status", "type_one", "type_two", "total_stats", "attack",
                             "defense", "special_attack", "special_defense", "speed"
                             "against_normal", "against_fire", "against_water", "against_electric", "against_grass",
                             "against_ice", "against_fight", "against_poison", "against_ground"
                             "against_flying", "against_psychic", "against_bug", "against_rock", "against_ghost",
                             "against_dragon", "against_dark", "against_steel", "against_fairy"), encoding='utf8')
exit()
pokemon_type = pokemon["type_one"]
pokemon_type_two = pokemon["type_two"]
pokemon_speed = pokemon["speed"]
total_stats = pokemon["total_stats"]
pokemon_gen = pokemon["gen"]
pokemon_name = pokemon['name']
pokemon_status = pokemon["status"]
pokemon_special_attack = pokemon["special_attack"]

max_idx = total_stats.idxmax()
max_index = pokemon_speed.idxmax()
best_pokemon = pokemon_name[max_idx]
min_idx = total_stats.idxmin()
worst_pokemon = pokemon_name[min_idx]

dark_pokemon = pokemon_name.where(pokemon_type == "Dark")
dark_pokemon = dark_pokemon.dropna()
dark_pokemon_stats = total_stats.where(pokemon_type == "Dark")
dark_pokemon_stats = dark_pokemon_stats.dropna()


# Try to load in the data from the file
X = pokemon[["against_normal", "against_fire", "against_water", "against_electric", "against_grass",
                             "against_ice", "against_fight", "against_poison", "against_ground"
                             "against_flying", "against_psychic", "against_bug", "against_rock", "against_ghost",
                             "against_dragon", "against_dark", "against_steel", "against_fairy"]]
y_classes = pokemon["type_one"]

# create the dictionary of total distribution
pokemon_types = []
center_value = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
explode = (0, 0, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
for p in pokemon_type:
    if p not in pokemon_types:
        pokemon_types.append(p)
pokemon_dict = dict(zip(pokemon_types, center_value))

for a in pokemon_type:
    if a in pokemon_dict:
        pokemon_dict[a] += 1
print(pokemon_dict)



names, counts = zip(*pokemon_dict.items())
plt.pie(counts, labels=names, explode=explode, shadow=True, startangle=90, autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'})
plt.title("Ratio of Each Pokemon Type")
plt.show()

# first plot describing my data
plt.bar(x=pokemon_type, height=pokemon_speed)
plt.title("Max Speed Depicted By each Type")
plt.ylabel("Max Speed stat by Type")
plt.xlabel("Pokemon type")
plt.xticks(rotation=50)
plt.show()

# second plot describing my data
plt.bar(x=pokemon_status, height=total_stats)
plt.title("The Total Stats Depicted by Pokemon Status")
plt.ylabel("The Max total Stats per Pokemon Status")
plt.xlabel("Pokemon Type")
plt.xticks(rotation=50)
plt.show()

# thirds plot describing my data
plt.boxplot(X)
plt.title("Box Plot depicting the Distribution of X")
plt.xlabel("Pokemon Stats")
plt.ylabel("The Distribution of each Column")
plt.show()

# fourth plot
plt.scatter(dark_pokemon_stats, dark_pokemon)
plt.title("The Contribution of each Dark type pokemon to the overall Stat Total of all Dark types")
plt.xlabel("Dark type pokemon")
plt.ylabel("The Contribution of each Dark type pokemon")
plt.show()

# create a label encoder, here we fit the code to have a good Y
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(y_classes)
y = label_encoder.transform(y_classes)

# fit an SVM classifier, then predict the classes for the entire dataset
model = sklearn.svm.SVC(kernel='rbf', class_weight='balanced')
model.fit(X, y)
y_predict = model.predict(X)


# grab the accuracy of y vs y_predicted
accuracy = sklearn.metrics.accuracy_score(y, y_predict)
balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y, y_predict)
print('accuracy: {:.1%}'.format(accuracy))
print('balanced accuracy: {:.1%}'.format(balanced_accuracy))

# find the confusion matrix of the y and y predicted
cm = sklearn.metrics.confusion_matrix(y, y_predict)
print('confusion matrix:')
print(cm)

# attempt one at stratified k-fold
#skf = StratifiedKFold(n_splits=5)
#for train_index, test_index in skf.split(X, y):
    #print("Train: ", train_index, "Test: ", test_index)
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]

sklearn.metrics.plot_confusion_matrix(model, X, y, cmap='Reds', display_labels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xticks(rotation=50)
plt.show()

# use stratified k-fold
accuracies = []
bal_accuracies = []
all_cms = None

skf_fold = sklearn.model_selection.StratifiedKFold(n_splits=5)
for i, (train_idx, test_idx) in enumerate(skf_fold.split(X, y)):
    print('\nfold {}'.format(i))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    # fit an SVM classifier, then predict the classes for the entire dataset
    model = sklearn.svm.SVC(class_weight='balanced')
    # use just the first two columns, to match the scatter plot
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # compute accuracy metrics for each fold
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)
    bal_accuracies.append(balanced_accuracy)
    print('accuracy: {:.1%}'.format(accuracy))
    print('balanced accuracy: {:.1%}'.format(balanced_accuracy))

    #confusion matrix for each fold
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    print('confusion matrix:')
    print(cm)

    # accumulate CMs across all folds for display at the end
    if all_cms is None:
        all_cms = np.zeros_like(cm)
    all_cms += cm

print()
print('mean accuracy: {:.2%}'.format(sum(accuracies) / len(accuracies)))
print('mean balanced accuracy: {:.2%}'.format(sum(bal_accuracies) / len(bal_accuracies)))
print('overall CM:')
print(all_cms)

# Let Sklearn do the plotting for me instead of manually printing the graph
sklearn.metrics.plot_confusion_matrix(model, X, y, cmap='Reds', display_labels=label_encoder.classes_)
plt.xticks(rotation=50)
plt.show()
#plt.figure()







