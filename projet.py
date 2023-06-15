from SPARQLWrapper import SPARQLWrapper, JSON
from owlready2 import *
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model

# Define the SPARQL query to retrieve animal information from Wikidata
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setQuery("""
    SELECT ?animal ?animalLabel ?imagenetID WHERE {
        VALUES ?animalLabel { "wolf"@fr "sharks"@fr "eagle"@fr "bear"@fr "indigo bunting"@fr "goldfish"@fr }
        ?animal rdfs:label ?animalLabel .
        OPTIONAL { ?animal wdt:P2671 ?imagenetID . }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],fr". }
    }
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

# Create a new ontology
onto = get_ontology("http://www.example.org/onto.owl")

# Define the namespaces
onto_namespace = onto.get_namespace("http://www.example.org/onto.owl#")
wd_namespace = Namespace("http://www.wikidata.org/entity/")

with onto:
    # Define the AnimalClass
    class AnimalClass(Thing):
        namespace = onto

    # Define the Image class
    class Image(Thing):
        namespace = onto

    # Define the Caractéristique_Morphologique class
    class Caractéristique_Morphologique(Thing):
        namespace = onto
    
    class contient(ObjectProperty):
        domain = [Image]
        range = [AnimalClass]

    # Define the subclasses for specific animals
    for animal_info in results["results"]["bindings"]:
        animal_name = animal_info["animal"]["value"].split("/")[-1]
        animal_class = types.new_class(animal_name, (AnimalClass,))
        animal_class.label = [animal_info["animalLabel"]["value"]]

    # Define the subclasses for specific Caractéristique Morphologique
    class Museau(Caractéristique_Morphologique):
        pass

    class Patte(Caractéristique_Morphologique):
        pass

    # Define the property 'possède'
    class possède(ObjectProperty):
        domain = [AnimalClass]
        range = [Caractéristique_Morphologique]

# Define the directory and retrieve image paths
root_directory = "animals"
image_paths = []

# os.walk returns a generator that creates a tuple of values
# for each directory in the directory tree
for dirpath, dirnames, filenames in os.walk(root_directory):
    # We are only interested in the .jpg files
    for filename in [f for f in filenames if f.endswith(".jpg")]:
        # os.path.join will create the full path to the image
        image_path = os.path.join(dirpath, filename)
        image_paths.append(image_path)

# Create instances of Image for each image path
with onto:
    for image_path in image_paths:
        Image(image_path.replace('\\', '/'))  # Create a new instance of Image for each image, ensure path format is universally valid

# Save the ontology to a file
onto.save(file="ontologie.owl", format="rdfxml")

# Load the populated ontology
onto = get_ontology("ontologie.owl").load()

# Define the classification algorithms
svm = SVC()
rf = RandomForestClassifier()
mlp = MLPClassifier()

# Retrieve the Image individuals from the ontology
image_individuals = list(onto.Image.instances())

def extract_features(image_path):
    # Load the InceptionV3 model (pre-trained on ImageNet)
    base_model = InceptionV3(weights='imagenet')
    
    # Create a new model without the top (fully connected) layers
    model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
    
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = preprocess_input(x)
    x = tf.expand_dims(x, axis=0)
    
    # Extract features from the image using the model
    features = model.predict(x)
    
    return features.flatten()


# Initialize empty lists to store features and labels
X_train = []
y_train = []

# Iterate over the training data
for image_individual in image_individuals:
    # Extract features from the image
    image_features = extract_features(image_individual.name)
    
    # Check if features were extracted
    if image_features is not None:
        # Append the features to X_train
        X_train.append(image_features)

        # Get the associated AnimalClass individuals
        animal_individuals = image_individual.contient

        # Iterate over each Animal individual
        for animal_individual in animal_individuals:
            # Check if the animal individual has a positive relation with the image individual
            if animal_individual.hasPositiveRelation.contains(image_individual):
                # Append "positive" label to y_train
                y_train.append("positive")
            else:
                # Append "negative" label to y_train
                y_train.append("negative")

# Convert the lists to numpy arrays for compatibility with scikit-learn classifiers
X_train = np.array(X_train)
y_train = np.array(y_train)

# Check if any features were extracted
if X_train.shape[0] == 0:
    print("No features extracted. Check the implementation of extract_features().")
else:
    # Reshape X_train
    n_samples, n_features = X_train.shape[0], X_train.shape[1]
    X_train = X_train.reshape(n_samples, n_features)

    # Train the classification algorithms using the training data
    svm.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    mlp.fit(X_train, y_train)

# After training, the classification algorithms are ready for prediction and classification tasks.
# Load the populated ontology

# Specify the path to the parent folder containing subfolders for each animal
parent_folder = "validation_images"

# Initialize an empty list to store the validation image paths
validation_images = []

# Iterate over the subfolders representing different animals
for animal_folder in os.listdir(parent_folder):
    # Get the path to the animal subfolder
    animal_folder_path = os.path.join(parent_folder, animal_folder)

    # Check if the subfolder is a directory
    if os.path.isdir(animal_folder_path):
        # Iterate over the files in the animal subfolder
        for file_name in os.listdir(animal_folder_path):
            # Get the path to each image file
            image_path = os.path.join(animal_folder_path, file_name)

            # Add the image path to the validation_images list
            validation_images.append(image_path)
onto = get_ontology("ontologie.owl").load()

# Iterate over the remaining validation images
for validation_image in validation_images:
    # Extract features from the validation image
    validation_features = extract_features(validation_image)

    # Predict the labels using the trained classification algorithms
    svm_label = svm.predict(validation_features)
    rf_label = rf.predict(validation_features)
    mlp_label = mlp.predict(validation_features)

    # Get the corresponding Image individual from the ontology
    image_individual = onto.Image(name=validation_image.filename)

    # Add the "positive" or "negative" relations to the animal individuals based on the predicted labels
    for animal_individual in onto.AnimalClass.instances():
        if svm_label == "positive":
            animal_individual.hasPositiveRelation.append(image_individual)
        else:
            animal_individual.hasNegativeRelation.append(image_individual)

        if rf_label == "positive":
            animal_individual.hasPositiveRelation.append(image_individual)
        else:
            animal_individual.hasNegativeRelation.append(image_individual)

        if mlp_label == "positive":
            animal_individual.hasPositiveRelation.append(image_individual)
        else:
            animal_individual.hasNegativeRelation.append(image_individual)

# Save the updated ontology with the added relations
onto.save(file="ontologie_updated.owl", format="rdfxml")

# Launch a reasoner to validate the coherence of the ontology
with onto:
    sync_reasoner()

# The reasoner will infer implicit relationships and check for logical consistency
# You can access the inferred relations and check for coherence or inconsistency
