import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.model_selection import train_test_split
class DataProcessor:
    def __init__(self, file_path="/Users/aditi/Desktop/CareCompanion/cleaned_medquad.csv"):
        self.df = pd.read_csv(file_path)
        self.clean_data()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['question'])

    def clean_data(self):
        print("\nMissing values:\n", self.df.isnull().sum())
        self.df = self.df[~self.df['answer'].str.contains(r'key points', case=False, na=False)]
        self.df['question'] = self.df['question'].str.lower().str.strip()
        self.df['answer'] = self.df['answer'].str.lower().str.strip()
        self.df = self.df.groupby("question", as_index=False).agg({
            "answer": lambda x: " || ".join(x.astype(str)),
            "source": lambda x: ", ".join(x.astype(str).unique())
        })
        self.df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=42)

class Chatbot:
    def __init__(self, data_processor):
        self.data_processor = data_processor

    def get_response(self, user_query):  # Renamed from get_best_match to get_response
        user_query = user_query.lower().strip()
        user_tfidf = self.data_processor.vectorizer.transform([user_query])
        similarities = cosine_similarity(user_tfidf, self.data_processor.tfidf_matrix)
        best_match_index = similarities.argmax()
        best_score = similarities[0, best_match_index]

        if best_score < 0.5:
            return "UNKNOWN", "Let me think some more.", "N/A"

        best_question = self.data_processor.df.iloc[best_match_index]['question']
        best_answer = self.data_processor.df.iloc[best_match_index]['answer'].replace(" || ", "\n- ")
        source = self.data_processor.df.iloc[best_match_index]['source']
        return best_question, best_answer, source

    def chat(self):
        print("Medical Chatbot is ready! Type 'exit' to stop.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                print("Chatbot: Goodbye! Take care!")
                break
            best_question, response, source = self.get_best_match(user_input)
            print(f"\nChatbot (matched question): {best_question}")
            print(f"Chatbot (answer): {response}")
            print(f"Source: {source}")

if __name__ == "__main__":
    data_processor = DataProcessor()
    chatbot = Chatbot(data_processor)
    chatbot.chat()
