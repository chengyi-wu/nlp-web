# coding: utf-8
from __future__ import print_function
from . import api
import json
import time
from flask import request
import os

from ..text_classifier.model import CnnModel, PSRModel

def init(base_dir):
    bizModel = CnnModel(
        vocab_dir = os.path.join(base_dir, 'business/vocab.txt'),
        categories = ['滴滴出行', '代驾', '快车', '出租车', '专车', '顺风车', '优步中国', '单车', '租车', '外卖', '优享', '街兔', '小巴']
    )
    bizModel.load(save_path = os.path.join(base_dir, 'business/textcnn/best_validation'))

    sentimentModel = CnnModel(
        vocab_dir = os.path.join(base_dir, 'sentiment/vocab.txt'),
        categories = ['正面', '负面', '中立']
    )
    sentimentModel.load(save_path = os.path.join(base_dir, 'sentiment/textcnn/best_validation'))

    positiveModel = CnnModel(
        vocab_dir = os.path.join(base_dir, 'positive/vocab.txt'),
        categories = ['商业相关', '司乘关系', '产品+业务', '科技品牌', '政策+行业']
    )
    positiveModel.load(save_path = os.path.join(base_dir, 'positive/textcnn/best_validation'))

    negativeModel = CnnModel(
        vocab_dir = os.path.join(base_dir, 'negative/vocab.txt'),
        categories = ['商业相关', '安全相关', '乘客体验', '司机相关', '政策压力']
    )
    negativeModel.load(save_path = os.path.join(base_dir, 'negative/textcnn/best_validation'))

    return {
        'business' : bizModel,
        'sentiment' : sentimentModel,
        'positive' : positiveModel,
        'negative' : negativeModel,
    }

models = init('app/text_classifier')
psrmodel = PSRModel()

# curl --header "Content-Type: application/json" --request POST \--data '{"content":"今晚21:20《滴滴快车·芝麻开门》三组选手闯关，精彩不停歇！“东北歌后”现场高歌，跑调、破音问题频出，“灵魂唱法”惊呆众人；从练习生到人气偶像，弟弟努力付出终有收获，现场帮姐姐圆梦；还有“火锅情侣”分享恋爱经历，男子花式送礼物，笑翻全场！更多精彩，尽在今晚21:20《滴滴快车·芝麻开门》，我们不见不散哦~\n\u3000\u3000“灵魂歌手”现场演唱惊呆众人\n\u3000\u3000陈立英自称“东北歌后”，她从小到大的梦想就是当一个歌手，唱出东北、唱出中国，唱到全世界！为此陈立英每天都在家练习唱歌，这让室友郝佳美备受折磨~\n\u3000\u3000来到《芝麻开门》陈立英便按捺不住内心的激动，现场一展歌喉，即便高音唱到失声，歌曲没有一句在调上，也没能打击到她的自信心！难怪在没有粉丝的情况下，她自己一人在直播间也能嗨唱了~正是这种“灵魂唱法”，让陈立英唱哭了幼儿园的小朋友，失去了工作；唱走了相亲对象，至今单身一人~\n\u3000\u3000而身为室友的郝佳美为了摆脱“煎熬”，决定为好友抢一套家庭影院回去，让她在家里好好练歌，不要再做“跑调歌后”了！\n\u3000\u3000干姐弟圆梦路上互相支持\n\u3000\u3000从话不会讲、歌不会唱、舞不会跳的练习生，到唱跳俱佳、舞台魅力四射的明日之星，奥斯丁为实现自己的梦想付出了许多努力！\n\u3000\u3000出道后的奥斯丁不仅是时尚界的宠儿，还经常参演综艺节目和电影，是万千粉丝心中的帅气偶像~而干姐姐李晨菲在自己逐梦过程中经常照顾和鼓励他，在遇到困难时，姐姐也无私地给予他帮助，这让奥斯丁一直心存感激。姐姐见证了他的成长，他也要陪姐姐一起圆梦！\n\u3000\u3000李晨菲一直想开一家属于自己的服装店，现在一切准备就绪，就差启动资金了~奥斯丁今晚就要帮姐姐抢一笔丰厚的资金，顺便帮她征婚，解决单身问题，究竟结果如何呢？\n\u3000\u3000“火锅情侣”恋爱经历笑翻全场\n\u3000\u3000知道女朋友喜欢吃火锅，罗雄飞在向袁梦影表白时，别出心裁地送了她一束肥牛卷花，成功牵手女神~之后罗雄飞又为她开了一家火锅店，高端宠溺女朋友，让袁梦影的一众姐妹羡慕不已~\n\u3000\u3000然而现在袁梦影却对火锅又爱又恨，罗雄飞开店之后大部分精力都倾注在了工作上，和自己的沟通变少了不说，而且每次送礼物都和火锅有关！继肥牛卷花之后，罗雄飞又送了她一束生菜西红柿花，能看又能吃，经济实惠！纪念日、七夕节各种节日都吃火锅庆祝，果然是“火锅直男”了！\n\u3000\u3000因工作而忘记了两人纪念日的罗雄飞，也认识到自己对女朋友的忽略，所以要为她抢一份东京之旅弥补自己的错误，顺便吃一顿海鲜火锅~究竟他们能否实现此行心愿呢？今晚21:20《滴滴快车·芝麻开门》，一起来看吧~\n\u3000\u3000这是一个有温度的公众号"}' http://localhost:5000/api/predict
@api.route('/classify', methods=['GET', 'POST'])
@api.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return json.dumps({
            'response' : 'OK'
        })
    data = request.get_json()
    # print(data)
    content = data['content']
    media = int(data['media'])
    duplicates = int(data['duplicates'])
    reads = int(data['reads'])
    businessline = models['business'].predict(content)
    sentiment = models['sentiment'].predict(content)
    tag = ''
    if sentiment == '正面':
        tag = models['positive'].predict(content)
    elif sentiment == '负面':
        tag = models['negative'].predict(content)
    else:
        tag = ''
    P = psrmodel.identify_propagation_level(media, duplicates, reads)
    S = 0
    R = 0

    if sentiment == '负面':
        S = psrmodel.identify_safety_level(content)
        R = psrmodel.identify_platform_level(content)

    psr = P * S * R

    severity = psrmodel.identify_severity(P, S, R)

    return json.dumps({
        'sentiment' : sentiment,
        'businessline' : businessline,
        'tag' : tag,
        'P' : P,
        'S' : S,
        'R' : R,
        'PSR' : psr,
        'severity' : severity
    }, ensure_ascii=False)

@api.route('/reload/<string:model_name>', methods=['GET'])
def reload(model_name):
    print(model_name)
    m = models.get(model_name)
    if m is not None:
        m.load(m.save_path)
        return json.dumps({
                'response' : 'OK'
            })
    return json.dumps({
                'response' : 'Not found'
            })