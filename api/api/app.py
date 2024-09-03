from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your dataset from the CSV file
csv_file_path = r"data/Book1.csv"
data = pd.read_csv(csv_file_path)

# Initialize the TfidfVectorizer and fit it on the existing student responses
tfidf_vectorizer = TfidfVectorizer()

def find_similar_responses_and_grade(new_response, question_id, data, tfidf_vectorizer,rubrics,correct_ans,max_response):
    # Filter the dataset to only include responses with the same QuestionID
    #print(new_response,question_id)
    filtered_data = data[data['QuestionID'] == question_id].copy()

    if filtered_data.empty:
        # Fit the TF-IDF vectorizer on the filtered correct responses
        tfidf_matrix = tfidf_vectorizer.fit_transform([correct_ans])
        # Transform the new response using the same TF-IDF vectorizer
        new_response_vector = tfidf_vectorizer.transform([new_response])
        
        # Compute cosine similarity between the new response and all filtered responses
        cosine_similarities = cosine_similarity(new_response_vector, tfidf_matrix).flatten()
        # Add the similarity scores to the filtered dataset
        filtered_data['CosineSimilarity'] = cosine_similarities
        assigned_grade=round(round(filtered_data.iloc[0]['CosineSimilarity'], 2)*max_response)
        save_new_data(question_id,new_response,max_response,assigned_grade,rubrics)
            
        return f"The new response is haveing a similarity of {round(filtered_data.iloc[0]['CosineSimilarity'], 2)}.  Assigned grade: {assigned_grade} "

    else:
        # Fit the TF-IDF vectorizer on the filtered student responses
        tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_data['StudentResponse'])

        # Transform the new response using the same TF-IDF vectorizer
        new_response_vector = tfidf_vectorizer.transform([new_response])

        # Compute cosine similarity between the new response and all filtered responses
        cosine_similarities = cosine_similarity(new_response_vector, tfidf_matrix).flatten()

        # Add the similarity scores to the filtered dataset
        filtered_data['CosineSimilarity'] = cosine_similarities

        # Sort the filtered data by similarity to find the most similar responses
        similar_responses = filtered_data.sort_values(by='CosineSimilarity', ascending=False)
        print(similar_responses.iloc[0]['CosineSimilarity'])
        # Check if the highest similarity is 100%
        
        if new_response==correct_ans:
            assigned_grade=max_response
            save_new_data(question_id,new_response,max_response,assigned_grade,rubrics)
            return f"The new response is identical to an correct. Assigned grade: {assigned_grade}"
        
        if round(similar_responses.iloc[0]['CosineSimilarity'], 2) == 1:
            assigned_grade = ((similar_responses.iloc[0]['StudentScore'])/(similar_responses.iloc[0]['MaxPossibleScore']))*max_response
            save_new_data(question_id,new_response,max_response,assigned_grade,rubrics)
            return f"The new response is identical to an existing one. Assigned grade: {assigned_grade}"
        elif round(similar_responses.iloc[0]['CosineSimilarity'], 2)>=0.75:
            assigned_grade = (((similar_responses.iloc[0]['StudentScore'])/(similar_responses.iloc[0]['MaxPossibleScore']))*max_response)*round(similar_responses.iloc[0]['CosineSimilarity'], 2)
            save_new_data(question_id,new_response,max_response,assigned_grade,rubrics)
            return f"The new response is having a similarity of {round(similar_responses.iloc[0]['CosineSimilarity'],2)}. Assigned grade: {assigned_grade}, rounded to {round(assigned_grade)}"
        elif round(similar_responses.iloc[0]['CosineSimilarity'], 2)<=0.75:
            # Fit the TF-IDF vectorizer on the filtered correct responses
            tfidf_matrix = tfidf_vectorizer.fit_transform([correct_ans])
            # Compute cosine similarity between the new response and all filtered responses
            cosine_similarities = cosine_similarity(new_response_vector, tfidf_matrix).flatten()
            # Add the similarity scores to the filtered dataset
            filtered_data['CosineSimilarity'] = cosine_similarities
            assigned_grade=round(round(filtered_data.iloc[0]['CosineSimilarity'], 2)*max_response)
            save_new_data(question_id,new_response,max_response,assigned_grade,rubrics)
            return f"The new response is haveing a similarity of {round(filtered_data.iloc[0]['CosineSimilarity'], 2)}.  Assigned grade: {assigned_grade} "

def save_new_data(question_id,new_response,max_response,assigned_grade,rubrics):
                # Create a new DataFrame with the new response
        # Question	QuestionID	Identifier	StudentResponse	CorrectAnswer	MaxPossibleScore	StudentScore	TenantName	Provider	Rubric	Grade	Subject
        # because some items are scalar, they have to be passed in [] to be converted to a list
    new_data = pd.DataFrame({
            "Question": ["FILLER"],
            "QuestionID": [question_id],
            "Identifier": ["FILLER"],
            "StudentResponse": [new_response],
            "CorrectAnswer": ["FILLER"],
            "MaxPossibleScore": [max_response],
            "StudentScore": [round(assigned_grade)],
            "TenantName": ["FILLER"],
            "Provider": ["FILLER"],
            "Rubric": [rubrics],
            "Grade": ["FILLER"],
            "Subject": ["FILLER"]
        })

        # Append the new data to the CSV file
    new_data.to_csv("data/Book1.csv", mode='a', header=False, index=False)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    json_data = request.get_json()
    student_response = json_data['studentResponse']
    question_id = int(json_data['questionId'])
    rubrics=json_data['rubrics']
    max_response=int(json_data['maxResponse'])
    correct_ans=json_data['correctAnswer']
    
    
    result = find_similar_responses_and_grade(student_response, question_id, data, tfidf_vectorizer,rubrics,correct_ans,max_response)

    if isinstance(result, str):
        score = None
        if 'Assigned grade' in result:
            score = result.split(': ')[-1]
        return jsonify({'message': result, 'score': score})
    else:
        return jsonify({'score': None, 'table': result.to_html(classes='data')})
    
@app.errorhandler(500)
def internal_server_error(e):
    return jsonify(error=str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)