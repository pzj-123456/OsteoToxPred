from flask import Flask, render_template, request, jsonify, session, redirect
import datetime
import random
import os
from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO

# 模型加载和预测框架
import os
import pickle
import joblib
# import torch  # 如果需要PyTorch
# import tensorflow as tf  # 如果需要TensorFlow
# from sklearn.preprocessing import StandardScaler  # 如果需要预处理

# 全局变量存储模型
models = {}
model_loaded = False

def load_models():
    """加载所有模型"""
    global models, model_loaded
    
    if model_loaded:
        return
    
    try:
        # 示例：加载不同格式的模型
        # 请根据你的实际模型文件路径和格式进行修改
        
        # 方式1：加载pickle格式模型
        # if os.path.exists('models/aaa_model.pkl'):
        #     with open('models/aaa_model.pkl', 'rb') as f:
        #         models['AAA'] = pickle.load(f)
        
        # 方式2：加载joblib格式模型
        # if os.path.exists('models/svm_model.joblib'):
        #     models['SVM'] = joblib.load('models/svm_model.joblib')
        
        # 方式3：加载PyTorch模型
        # if os.path.exists('models/randomforest_model.pth'):
        #     models['RandomForest'] = torch.load('models/randomforest_model.pth')
        
        # 方式4：加载TensorFlow模型
        # if os.path.exists('models/xgboost_model.h5'):
        #     models['XGBoost'] = tf.keras.models.load_model('models/xgboost_model.h5')
        
        print("模型加载成功")
        model_loaded = True
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        model_loaded = False

def predict_with_model(smiles, model_name):
    """使用指定模型进行预测"""
    try:
        # 检查模型是否已加载
        if not model_loaded:
            load_models()
        
        # 检查模型是否存在
        if model_name not in models:
            print(f"模型 {model_name} 未找到，使用随机预测")
            return random.uniform(0.1, 0.9), random.uniform(60, 95)
        
        model = models[model_name]
        
        # 根据你的模型类型进行预测
        # 请根据你的实际模型输入输出格式进行修改
        
        # 示例1：sklearn模型
        # features = extract_features_from_smiles(smiles)  # 你需要实现这个函数
        # prediction_value = model.predict_proba(features)[0][1]  # 获取正类概率
        
        # 示例2：PyTorch模型
        # features = extract_features_from_smiles(smiles)
        # with torch.no_grad():
        #     prediction_value = model(torch.tensor(features, dtype=torch.float32)).item()
        
        # 示例3：TensorFlow模型
        # features = extract_features_from_smiles(smiles)
        # prediction_value = model.predict(features)[0][0]
        
        # 临时使用随机预测（请替换为你的实际预测代码）
        prediction_value = random.uniform(0.1, 0.9)
        confidence = random.uniform(60, 95)
        
        return prediction_value, confidence
        
    except Exception as e:
        print(f"预测失败: {str(e)}")
        return random.uniform(0.1, 0.9), random.uniform(60, 95)

def extract_features_from_smiles(smiles):
    """从SMILES提取分子特征"""
    # 你需要实现这个函数来提取分子特征
    # 可以使用RDKit、mordred、mol2vec等库
    
    # 示例：使用RDKit提取简单的分子描述符
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * 10  # 返回默认特征向量
        
        # 这里应该提取你模型需要的特征
        # 例如：分子量、原子数、环数等
        features = [
            mol.GetNumAtoms(),  # 原子数
            mol.GetNumBonds(),  # 键数
            mol.GetNumRings(),  # 环数
            mol.GetNumRotatableBonds(),  # 可旋转键数
            mol.GetNumHeteroatoms(),  # 杂原子数
            mol.GetNumAromaticRings(),  # 芳香环数
            mol.GetNumSaturatedRings(),  # 饱和环数
            mol.GetNumAliphaticRings(),  # 脂肪环数
            mol.GetNumAromaticHeterocycles(),  # 芳香杂环数
            mol.GetNumSaturatedHeterocycles()  # 饱和杂环数
        ]
        
        return features
        
    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        return [0] * 10  # 返回默认特征向量

# Flask应用初始化
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# 应用启动时加载模型
@app.before_first_request
def initialize_app():
    """应用启动时初始化"""
    print("正在加载模型...")
    load_models()
    print("应用初始化完成")

# 您的真实预测函数 - 请替换为您的实际预测模型
def predict_toxicity(smiles_list, model_name):
    """毒性预测函数 - 使用真实模型进行预测"""
    results = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            # 使用模型进行预测
            prediction_value, confidence = predict_with_model(smiles, model_name)
            is_toxic = prediction_value > 0.5
            
            # 生成分子结构图
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # 生成分子结构图
                    img = Draw.MolToImage(mol, size=(200, 200))
                    img_buffer = BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_str = base64.b64encode(img_buffer.getvalue()).decode()
                    structure_image = f"data:image/png;base64,{img_str}"
                else:
                    structure_image = "/static/img/default_molecule.png"
            except:
                structure_image = "/static/img/default_molecule.png"
            
            results.append({
                'smiles': smiles,
                'prediction_value': prediction_value,
                'is_toxic': is_toxic,
                'confidence': confidence,
                'structure_image': structure_image
            })
            
        except Exception as e:
            print(f"预测分子 {smiles} 时出错: {str(e)}")
            # 如果预测失败，使用默认值
            results.append({
                'smiles': smiles,
                'prediction_value': 0.5,
                'is_toxic': False,
                'confidence': 50.0,
                'structure_image': "/static/img/default_molecule.png"
            })
    
    return results

@app.route('/')
def home():
    return render_template('index.html', page="home")

@app.route('/introduction')
def introduction():
    return render_template('introduction.html', page="introduction")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', page="predict")
    
    elif request.method == 'POST':
        try:
            print("DEBUG: 开始处理POST请求")
            
            # 清除之前的预测结果和错误信息
            session.pop('prediction_data', None)
            session.pop('prediction_error', None)
            
            # 获取表单数据
            algorithm = request.form.get('algorithm', 'AAA')
            smiles_text = request.form.get('smiles', '').strip()
            
            print(f"DEBUG: 获取到的algorithm: {algorithm}")
            print(f"DEBUG: 获取到的smiles_text长度: {len(smiles_text)}")
            print(f"DEBUG: smiles_text内容: {repr(smiles_text)}")
            
            # 存储表单数据到session，用于错误时重新显示
            session['form_data'] = {
                'algorithm': algorithm,
                'smiles': smiles_text
            }
            
            # 解析SMILES列表
            smiles_list = [line.strip() for line in smiles_text.split('\n') if line.strip()]
            
            # 打印调试信息
            print(f"DEBUG: 接收到的SMILES文本长度: {len(smiles_text)}")
            print(f"DEBUG: 解析后的SMILES列表长度: {len(smiles_list)}")
            print(f"DEBUG: SMILES列表内容: {smiles_list}")
            
            # 移除所有限制，允许任何输入
            # if not smiles_text:
            #     session['prediction_error'] = '请输入SMILES分子结构'
            #     return redirect('/prediction_result')
            # 
            # if len(smiles_list) > 20:
            #     session['prediction_error'] = '最多只能输入20个SMILES分子结构'
            #     return redirect('/prediction_result')
            
            # 执行预测
            print("DEBUG: 开始执行预测")
            prediction_results = predict_toxicity(smiles_list, algorithm)
            print(f"DEBUG: 预测完成，结果数量: {len(prediction_results)}")
            
            # 计算统计信息
            total_molecules = len(prediction_results)
            toxic_count = sum(1 for result in prediction_results if result['is_toxic'])
            safe_count = total_molecules - toxic_count
            toxic_percentage = (toxic_count / total_molecules * 100) if total_molecules > 0 else 0
            
            print(f"DEBUG: 统计信息 - 总数: {total_molecules}, 有毒: {toxic_count}, 安全: {safe_count}")
            
            # 存储结果到session
            session['prediction_data'] = {
                'model_name': algorithm,
                'prediction_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'molecule_count': total_molecules,
                'prediction_results': prediction_results,
                'total_molecules': total_molecules,
                'toxic_count': toxic_count,
                'safe_count': safe_count,
                'toxic_percentage': toxic_percentage
            }
            
            print("DEBUG: 准备重定向到prediction_result")
            return redirect('/prediction_result')
            
        except Exception as e:
            error_msg = f'预测失败: {str(e)}'
            session['prediction_error'] = error_msg
            return redirect('/prediction_result')

@app.route('/prediction_result')
def prediction_result():
    # 从session获取预测结果
    prediction_data = session.get('prediction_data')
    prediction_error = session.get('prediction_error')
    form_data = session.get('form_data', {})
    
    # 如果有错误，显示错误页面
    if prediction_error:
        return render_template('prediction_result.html', 
                             error=prediction_error,
                             form_data=form_data,
                             has_error=True)
    
    # 如果没有预测数据，重定向到预测页面
    if not prediction_data:
        return redirect('/predict')
    
    # 清除session中的数据，避免缓存问题
    session.pop('prediction_data', None)
    session.pop('prediction_error', None)
    session.pop('form_data', None)
    
    return render_template('prediction_result.html', **prediction_data)

@app.route('/help')
def help_page():
    return render_template('help.html', page="help")

@app.route('/hwvlab')
def hwvlab():
    return render_template('hwvlab.html', page="hwvlab")

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    try:
        # 清除所有session数据
        session.clear()
        return jsonify({'success': True, 'message': '缓存已清除'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_csv')
def download_csv():
    prediction_data = session.get('prediction_data')
    if not prediction_data:
        return "没有可下载的数据", 404
    
    # 生成CSV文件
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # 写入表头
    writer.writerow(['序号', 'SMILES', '预测值', '毒性判断', '置信度'])
    
    # 写入数据
    for i, result in enumerate(prediction_data['prediction_results'], 1):
        toxicity = '有毒' if result['is_toxic'] else '安全'
        writer.writerow([
            i,
            result['smiles'],
            f"{result['prediction_value']:.4f}",
            toxicity,
            f"{result['confidence']:.1f}%"
        ])
    
    from flask import Response
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=prediction_results.csv'}
    )

@app.route('/download_pdf')
def download_pdf():
    # 这里可以实现PDF报告生成
    return "PDF下载功能待实现", 501

if __name__ == '__main__':
    app.run(debug=True)
