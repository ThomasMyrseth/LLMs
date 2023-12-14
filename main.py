from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from Read_pdf import Read_pdf #from filename import class
from Search_web import Search_web

"""
add conversation memory
allow for use of multiple PDFs
change model used in llm to a cheaper one
"""

class main:
    def __init__(self):
        self.pdf_titles = []
        self.dbs = []
        self.pdf_reader = Read_pdf()
        self.web_searcher = Search_web()

    def add_new_pdf(self, pdf_title):
        if pdf_title in self.pdf_titles:
            string = pdf_title, " is already uploaded!"
            return string
        
        self.pdf_titles.append(pdf_title)
        db = self.pdf_reader.upload_and_create_vectorstore(pdf_title)
        self.dbs.append(db)

    def query_web(self, query):
        response = self.web_searcher.search_query(query)
        return response

    def query_pdf(self, pdf_title, query):

        if pdf_title not in self.pdf_titles:
            return "You must first upload: ", pdf_title
        
        response, docs = self.pdf_reader.main(pdf_title=pdf_title, query=query)
        return response, docs

    def query_web_and_pdf(self, pdf_title, query):
        llm = OpenAI(temperature=0)

        if pdf_title not in self.pdf_titles:
            db = self.pdf_reader.upload_and_create_vectorstore(pdf_title=pdf_title)
            self.dbs.append(db)
            self.pdf_titles.append(pdf_title)

        template = PromptTemplate(
            input_variables=["content", "query"],
            template="""You are a financial analyst focusing on quantitative values. Please elaborate on your answers where possible. Use numbers where possible.
                        Given this information: {content}
                        Answer: {query}
                        """
        )

        content = ""
        content += self.query_web(query=query)
        response, docs = self.query_pdf(pdf_title=pdf_title, query=query)
        content += response["content"]

        chain = LLMChain(llm=llm, prompt=template)
        response = chain.run({"content": content, "query": query})

        return response, docs
    

test = main()
pdf_title = "raw_data/AkerBP_q1_oppsumering_TABELL.pdf"
query = "Whats the EBITDA margin for 2023 and 2022"
query2 = "Give me the key factors going into 2030 AkerBP should be conserned about"

test.add_new_pdf(pdf_title=pdf_title)
#pdf_response = test.query_pdf(pdf_title=pdf_title, query=query)
#print(pdf_response)

#web_response = test.query_web(query=query)
#print(web_response)

web_and_pdf_response, docs = test.query_web_and_pdf(pdf_title=pdf_title, query=query2)
print(web_and_pdf_response)
print("\n\n")
print( set([doc.metadata["page_num"] for doc in docs]) )
        

