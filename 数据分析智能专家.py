import warnings
warnings.filterwarnings('ignore')
from langchain.agents import *
from langchain.tools import Tool,tool
from langchain_experimental.agents import *

from langchain_openai import ChatOpenAI
def LLM(model='qwen-max'):
    llm=ChatOpenAI(base_url='YOUR BASE URL',api_key='YOUR API_KEY',model=model,temperature=0)
    return llm

import warnings
warnings.filterwarnings('ignore')
from langchain.agents import *
from langchain.tools import Tool,tool
from langchain_experimental.agents import *
llm=LLM()

from langchain_experimental.tools.python.tool import PythonAstREPLTool
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
llm=LLM(model='qwen-coder-plus-latest')
from langchain import hub
from langchain.agents import *
from langchain.tools import Tool,tool
from langchain.utilities import *
from langchain_experimental.agents import *
from langchain.prompts import *
from langchain.schema import *
from langchain.agents.agent import *
from langchain.output_parsers import *
class pandas_agent:
    def __init__(self,df:pd.DataFrame):
        self.df_locals={'df':df}
        self.agent=create_pandas_dataframe_agent(llm=llm,df=df,allow_dangerous_code=True,agent_type='tool-calling',verbose=True,return_intermediate_steps=True)
        content=self.agent.agent.runnable.middle[0].messages[0].content
        self.agent.tools[0]
        self.agent.tools[0]._run('''import matplotlib.pyplot as plt  
import matplotlib  
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False ''')
        first=content.index('This')
        self.agent.agent.runnable.middle[0].messages[0].content='''你是一个中文版的数据分析智能专家,熟练使用pandas,numpy,seaborn,scikit-learn,scipy,sympy等库,
        你会使用一个python代码编辑器工具进行代码的生成与运行以满足用户的问答需求,你有以下几点需要特别注意!!!\n
        1.用户的输入中会告诉你之前你自己生成了哪些代码命名变量(以python的字典格式说明)，从而你可以直接利用这些变量进行后续的代码操作，例如:"tree:运用sklearn在df数据上进行决策树训练的决策树变量"
        2.你生成的代码应该包含一种注释(必须是中文)，专门表明你生成的代码的自定义变量名称对应的实际含义
        3.你会进行机器学习与数据挖掘的工作，并且默认df的最后一列为目标变量，其余为特征变量
        4.在生成的代码中不要重复命名自定义变量！！！
        5.用户的输入中会说明你和用户之前的聊天记录，以便更好得理解用户接下来的话\n\n
        6.如何需要运用seaborn，matplotlib等库进行数据的可视化展示时，请将要绘制的图形的自变量名称统一命名为image
        ，这里可以不用考虑自定义变量名称的重复,并且不需要亲自绘制出来，只需要用户获取image这个变量即可
        7.如果用户没有明确得说明要绘制统计图形，你就千万不要在代码中去生成绘制图形的代码
        '''+content[first:]
        self.code_entity=''
        self.history=''
        response_schemas=[ResponseSchema(name='变量描述',description='用来描述代码中每个变量的含义')]
        output_parser=StructuredOutputParser.from_response_schemas(response_schemas=response_schemas)
        format_instructions=output_parser.get_format_instructions()
        get_code_entity_prompt=PromptTemplate.from_template(template='''你是一个提取python代码中自定义变量实体的智能助手，
        擅长从代码中提取自定义变量实体，用户会提供给你一些代码以及默认的代码变量，你需要根据这些来提取新的代码变量,
        请根据用户提供给你的如下代码进行变量实体的解析与提取:\n{code}\n\n
        以下是你输出内容的格式要求\n:{format_instructions}
        ''',partial_variables={"format_instructions": format_instructions})
        self.get_code_entity_llm=get_code_entity_prompt|LLM(model='qwen-max')
        self.code_history=''
        is_show_prompt=PromptTemplate.from_template(template='''你正在辅助另外一个python代码生成智能助手,
                                                    根据用户的输入来推测代码生成助手是否需要运用seaborn，matplotlib等库进行
                                                    数据可视化展示,只有用户在输入中明确显式得指示了要绘制统计图形才能
                                                    认为需要进行数据可视化展示，如果需要就只需要返回"是",否则只需要返回"否".
                                                    千万不要输出除了"是"或者"否"这两个字以外的其他内容\n
                                                    用户的输入为:\n{input}''')
        self.is_show_llm=is_show_prompt|LLM(model='qwen-max')|BooleanOutputParser(true_val='是',false_val='否')
    def run(self,query:str):
        is_show=self.is_show_llm.invoke({'input':query})
        code='这是你之前生成的代码中的每一个自定义变量名称以及含义:\n{code_mess}'.format(code_mess=str(self.code_entity))
        chat_history='这是之前的聊天历史记录{history}'.format(history='\n'.join(self.history))
        self.history+=str({'user':query})
        query=code+'\n'+chat_history+'\n'+query
        res=self.agent.invoke({'input':query})
        if not is_show:
            if res['intermediate_steps']:
                code=res['intermediate_steps'][0][0].tool_input['query']
                code_entity=self.get_code_entity_llm.invoke({'code':code})
                first,end=code_entity.content.index('{'),code_entity.content.index('}')
                self.code_entity+=code_entity.content[first:end+1]+'\n'
                self.code_history+=code+'\n'
            output=res['output']
            self.history+=str({'ai',output})
            return res
        image=self.agent.tools[0].locals['image']
        return image
    def clear_history(self):
        self.code_history=''
        self.history=''

import gradio as gr
import seaborn as sns
import matplotlib.pyplot as plt
df=None
with gr.Blocks(theme='black') as demo:
    with gr.Tab(label='上传csv数据',id=0) as show_data:
        agent:pandas_agent=llm
        def upload_csv(flie):
            global df,agent
            df=pd.read_csv(flie.name).iloc[:,1:]
            agent=pandas_agent(df=df)
            return gr.update(visible=True,value=df)
        upload=gr.File(visible=True)
        df_gradio=gr.DataFrame(visible=False)
        upload.upload(fn=upload_csv,inputs=upload,outputs=df_gradio)
    with gr.Tab(label='数据可视化区',id=2) as data_show:
        image=gr.Image()
    with gr.Tab(label='数据分析处理',id=1) as data_analys:
        with gr.Row() as chat_code:
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label='数据科学智能助手',height=500,type='messages',scale=7)
                with gr.Row():
                    input=gr.Textbox(scale=3,interactive=True)
                    send=gr.Button(value='发送',scale=1)
            with gr.Column(scale=3):
                code=gr.Code(label='代码展示区',language='python',lines=20,scale=3,min_width=240)
                clear=gr.Button(value='清除')
        history,generate_code = [],''
        def response(query:str):
            global history,agent,res,generate_code
            res=agent.run(query)
            if type(res)!=dict:
                try:
                    image_show=res 
                    image_show.savefig('image.png')
                except AttributeError:
                    fig=res.get_figure()
                    fig.savefig('image.png')
                res='绘制的图形已完成，请到可视化区查看'
            else:
                if res['intermediate_steps']:
                    generate_code+='\n'+res['intermediate_steps'][0][0].tool_input['query']
                res=res['output']
            history+=[{'role':'user','content':query},{'role':'assistant','content':res}]
            return history,gr.Code(value=generate_code,label='代码展示区',language='python',interactive=True),gr.Image(value=plt.imread(fname='image.png'))
        send.click(fn=response,inputs=input,outputs=[chatbot,code,image])
        clear.click(fn=lambda x:gr.Code(value='',label='代码展示区',language='python',interactive=True),outputs=code)

demo.launch()
