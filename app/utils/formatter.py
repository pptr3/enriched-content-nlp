class Formatter():
    def __init__(self) -> None:
        pass

    def refactor_website_content_format(self, website_content):
        '''
        type(website_content) == {'Name1': 'link1', 'Name2': 'link2'}
        type(out_list) == list[{"name": Name1, "url" : link1}, {"name": Name2, "url" : link2}]
        '''
        out_list = []
        if not website_content:
            return out_list
        for name, link in website_content.items():
            out_list.append({'name': name, 'url': link})
        return out_list
    
    def refactor_keywords_format(self, keywords):
        '''
        type(keywords) == [['wrestler', 0.5038, False], ['rapper', 0.4057, False]]
        type(out_keywords) == list[{"keyword" : wrestler, "score" : 0.5038}, {"keyword" : rapper, "score" : 0.4057}]
        '''
        out_keywords = []
        if not keywords:
            return out_keywords
        for el in keywords:
            out_keywords.append({'keyword': el[0], 'score': el[1]})
        return out_keywords
    
    def refactor_topic_format(self, topic):
        '''
        type(topic) == ('wrestler', 0.257540762424469)
        type(out_topic) == list[{"topic" : wrestler, "score" : 0.257540762424469}]
        '''
        if not topic:
            return  []
        return  [{"topic" : topic[0], "score" : topic[1]}]
    
    def refactor_faq_format(self, faq):
        '''
        if `faq` is not empty, it will be == [{"question": "question1", "answer": "answer1"}]
        '''
        return [] if faq[0]["question"] == "" else faq