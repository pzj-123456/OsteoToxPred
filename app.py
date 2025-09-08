from flask import Flask, render_template, request, jsonify, session, redirect
import datetime
import random
import os
from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO
from flask_session import Session
import redis

# 强制设置matplotlib使用非交互式后端，避免Tkinter线程冲突
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# 配置session使用Redis存储，解决cookie过大的问题
app.config['SESSION_TYPE'] = 'redis'
try:
    # 尝试连接Redis
    redis_client = redis.from_url('redis://localhost:6379')
    redis_client.ping()  # 测试连接
    app.config['SESSION_REDIS'] = redis_client
    print("成功连接到Redis服务器")
except:
    # 如果Redis不可用，回退到文件系统存储
    print("Redis连接失败，使用文件系统存储session")
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = './flask_session'
    # 确保目录存在
    os.makedirs('./flask_session', exist_ok=True)

# 初始化session
Session(app)

# 配置临时图片目录
from pathlib import Path
import uuid

TMP_IMG_DIR = Path(app.root_path) / "static" / "tmp"
TMP_IMG_DIR.mkdir(parents=True, exist_ok=True)

def save_mol_image(mol, job_id, idx):
    """保存分子结构图为PNG文件，返回静态文件路径"""
    try:
        img = Draw.MolToImage(mol, size=(200, 200))
        filename = f"{job_id}_{idx}.png"
        path = TMP_IMG_DIR / filename
        img.save(str(path))
        return f"/static/tmp/{filename}"
    except Exception as e:
        print(f"保存分子图片失败: {e}")
        return "/static/img/default_molecule.png"

def cleanup_old_images():
    """清理超过1小时的旧图片文件"""
    try:
        print(f"DEBUG: 开始清理旧图片，目录: {TMP_IMG_DIR}")
        
        if not TMP_IMG_DIR.exists():
            print(f"DEBUG: 临时图片目录不存在: {TMP_IMG_DIR}")
            return
        
        current_time = datetime.datetime.now()
        print(f"DEBUG: 当前时间: {current_time}")
        
        # 获取所有PNG文件
        png_files = list(TMP_IMG_DIR.glob("*.png"))
        print(f"DEBUG: 找到 {len(png_files)} 个PNG文件")
        
        cleaned_count = 0
        for img_file in png_files:
            try:
                # 获取文件修改时间
                mtime = datetime.datetime.fromtimestamp(img_file.stat().st_mtime)
                file_age = current_time - mtime
                age_seconds = file_age.total_seconds()
                
                # print(f"DEBUG: 文件 {img_file.name} 年龄: {age_seconds:.1f} 秒")
                
                if age_seconds > 600:  # 10min        
                    img_file.unlink()
                    # print(f"DEBUG: 清理旧图片文件: {img_file}")
                    cleaned_count += 1
                
                    
            except Exception as e:
                print(f"DEBUG: 处理文件 {img_file} 时出错: {e}")
                continue
        
        print(f"DEBUG: 清理完成，共清理了 {cleaned_count} 个文件")
        
    except Exception as e:
        print(f"DEBUG: 清理旧图片文件失败: {e}")
        import traceback
        traceback.print_exc()

# 导入毒性预测模型
try:
    from models.model_wrapper import ToxicityPredictor
    # 初始化模型（全局变量，避免重复加载）
    toxicity_predictor = None
    def get_predictor():
        global toxicity_predictor
        if toxicity_predictor is None:
            toxicity_predictor = ToxicityPredictor()
        return toxicity_predictor
    print("毒性预测模型导入成功")
except ImportError as e:
    print(f"毒性预测模型导入失败: {e}")
    toxicity_predictor = None

# 初始化预测器缓存
toxicity_predictors = {}

def get_predictor(model_name: str):
    """按名称懒加载并缓存预测器对象"""
    from models.sklearn_predictor import SklearnPredictor  # 延迟导入，避免循环
    global toxicity_predictors
    if model_name in toxicity_predictors:
        return toxicity_predictors[model_name]
    if model_name.lower() in ("btp-mffgnn", "btp-mffgnn_model"):
        predictor = ToxicityPredictor()
    elif model_name.lower() in ("svm", "svm_model"):
        predictor = SklearnPredictor("SVM_model")
    elif model_name.lower() in ("randomforest", "randomforest_model"):
        predictor = SklearnPredictor("RandomForest_model")
    elif model_name.lower() in ("xgboost", "xgboost_model"):
        predictor = SklearnPredictor("XGBoost_model")
    else:
        raise ValueError(f"未知模型: {model_name}")
    toxicity_predictors[model_name] = predictor
    return predictor

# 毒性预测函数 - 使用真实的GINet模型
def predict_toxicity(smiles_list, model_name):
    """使用GINet模型进行毒性预测"""
    results = []
    
    # 生成唯一的任务ID
    job_id = str(uuid.uuid4())[:8]
    
    # 清理旧图片
    cleanup_old_images()

    
    try:
        # 获取模型预测器
        predictor = get_predictor(model_name)
            
        # 使用模型进行预测
        predictions = predictor.predict_batch(smiles_list)
        
        for i, (smiles, pred_result) in enumerate(zip(smiles_list, predictions)):
            # 从预测结果中提取数据
            prediction_value = pred_result['toxicity_probability']
            is_toxic = prediction_value > 0.5
            confidence = prediction_value * 100  # 转换为百分比
            
            # 生成分子结构图并保存为静态文件
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # 保存图片并获取路径
                    structure_image = save_mol_image(mol, job_id, i)
                else:
                    structure_image = "/static/img/default_molecule.png"
            except Exception as e:
                print(f"处理SMILES失败: {smiles}, 错误: {e}")
                structure_image = "/static/img/default_molecule.png"
            
            # 创建注意力权重可视化（仅当当前预测器支持时）
            attention_vis = None
            try:
                if hasattr(predictor, 'create_attention_visualization'):
                    attention_vis = predictor.create_attention_visualization(
                        smiles,
                        pred_result.get('attention_weights', []),
                        job_id,
                        i
                    )
            except Exception as e:
                print(f"创建注意力可视化失败: {e}")
                attention_vis = None
            
            results.append({
                'smiles': smiles,
                'prediction_value': prediction_value,
                'is_toxic': is_toxic,
                'confidence': confidence,
                'structure_image': structure_image,
                'non_toxic_probability': 1 - prediction_value,
                'attention_weights': pred_result.get('attention_weights', []),
                'attention_visualization': attention_vis  # 添加注意力可视化
            })
            
    except Exception as e:
        print(f"模型预测失败: {e}")
        # 不再使用随机预测，直接抛出异常
        raise Exception(f"模型 {model_name} 预测失败: {str(e)}")
    
    return results

def validate_smiles(smiles):
    """验证SMILES分子是否有效"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "无效的SMILES格式"
        
        # 检查分子是否为空
        if mol.GetNumAtoms() == 0:
            return False, "分子不包含任何原子"
        
        # 检查分子是否过于简单（比如只有单个原子）
        if mol.GetNumAtoms() < 2:
            return False, "分子过于简单，至少需要2个原子"
        
        # 检查分子是否包含无效的化学键
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            return False, f"分子化学结构无效: {str(e)}"
        
        return True, "有效分子"
        
    except Exception as e:
        return False, f"验证失败: {str(e)}"

def validate_smiles_list(smiles_list):
    """验证SMILES列表，返回有效和无效的分子"""
    valid_smiles = []
    invalid_smiles = []
    
    for smiles in smiles_list:
        is_valid, message = validate_smiles(smiles)
        if is_valid:
            valid_smiles.append(smiles)
        else:
            invalid_smiles.append({
                'smiles': smiles,
                'error_message': message
            })
    
    return valid_smiles, invalid_smiles

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
            algorithm = request.form.get('algorithm', 'BTP-MFFGNN')
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
            
            # 验证SMILES分子
            valid_smiles, invalid_smiles = validate_smiles_list(smiles_list)
            
            print(f"DEBUG: 有效分子数量: {len(valid_smiles)}")
            print(f"DEBUG: 无效分子数量: {len(invalid_smiles)}")
            
            # 检查是否有有效分子
            if not valid_smiles:
                session['prediction_error'] = '没有找到有效的分子结构，请检查输入的SMILES格式'
                return redirect('/prediction_result')
            
            # 执行预测（只对有效分子）
            print("DEBUG: 开始执行预测")
            prediction_results = predict_toxicity(valid_smiles, algorithm)
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
                'toxic_percentage': toxic_percentage,
                'invalid_molecules': invalid_smiles,  # 添加无效分子信息
                'total_input_molecules': len(smiles_list),  # 总输入分子数
                'valid_molecules_count': len(valid_smiles)  # 有效分子数
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
        print("DEBUG: 没有预测数据，重定向到预测页面")
        return redirect('/predict')
    print(f"DEBUG: 有预测数据，显示预测结果页面{prediction_data}")
    
    # 清除session中的数据，避免缓存问题
    session.pop('prediction_data', None)
    session.pop('prediction_error', None)
    session.pop('form_data', None)
    
    return render_template('prediction_result.html', **prediction_data)

@app.route('/health')
def health_check():
    """健康检查端点"""
    try:
        # 检查模型是否加载
        predictor = get_predictor('BTP-MFFGNN')
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.datetime.now().isoformat(),
            'model_loaded': True
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.datetime.now().isoformat(),
            'error': str(e)
        }), 500

@app.route('/help')
def help_page():
    return render_template('help.html', page="help")

@app.route('/hwvlab')
def hwvlab():
    return render_template('hwvlab.html', page="hwvlab")

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """清理缓存数据"""
    try:
        print("DEBUG: 开始清理缓存...")
        
        # 清理session数据
        session.clear()
        print("DEBUG: Session数据已清理")
        
        # 清理临时图片文件
        cleanup_old_images()
        print("DEBUG: 临时图片清理完成")
        
        return jsonify({'success': True, 'message': '缓存清理成功'})
    except Exception as e:
        print(f"DEBUG: 清理缓存失败: {e}")
        return jsonify({'success': False, 'message': f'清理失败: {str(e)}'})

@app.route('/cleanup_images', methods=['POST'])
def cleanup_images():
    """手动清理临时图片文件"""
    try:
        print("DEBUG: 开始清理临时图片...")
        
        # 检查目录是否存在
        if not TMP_IMG_DIR.exists():
            print(f"DEBUG: 临时图片目录不存在: {TMP_IMG_DIR}")
            return jsonify({'success': False, 'message': '临时图片目录不存在'})
        
        # 统计清理前的文件数量
        before_count = len(list(TMP_IMG_DIR.glob("*.png")))
        print(f"DEBUG: 清理前有 {before_count} 个PNG文件")
        
        # 清理旧图片
        cleanup_old_images()
        
        # 统计清理后的文件数量
        after_count = len(list(TMP_IMG_DIR.glob("*.png")))
        cleaned_count = before_count - after_count
        print(f"DEBUG: 清理后剩余 {after_count} 个PNG文件，清理了 {cleaned_count} 个文件")
        
        return jsonify({
            'success': True, 
            'message': f'图片清理成功！清理了 {cleaned_count} 个文件',
            'cleaned_count': cleaned_count
        })
    except Exception as e:
        print(f"DEBUG: 清理图片失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'图片清理失败: {str(e)}'})

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
    writer.writerow(['序号', 'SMILES', '预测值', '毒性判断', '置信度', '状态'])
    
    # 写入有效分子数据
    for i, result in enumerate(prediction_data['prediction_results'], 1):
        toxicity = '有毒' if result['is_toxic'] else '安全'
        writer.writerow([
            i,
            result['smiles'],
            f"{result['prediction_value']:.4f}",
            toxicity,
            f"{result['confidence']:.1f}%",
            '有效'
        ])
    
    # 写入无效分子数据
    invalid_molecules = prediction_data.get('invalid_molecules', [])
    for i, invalid_mol in enumerate(invalid_molecules, len(prediction_data['prediction_results']) + 1):
        writer.writerow([
            i,
            invalid_mol['smiles'],
            'N/A',
            'N/A',
            'N/A',
            f"无效: {invalid_mol['error_message']}"
        ])
    
    from flask import Response
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=prediction_results.csv'}
    )

@app.route('/download_dataset')
def download_dataset():
    """下载骨毒性数据集文件"""
    try:
        dataset_path = os.path.join(app.root_path, 'data', 'bonetox-new.csv')
        
        if not os.path.exists(dataset_path):
            return "数据集文件不存在", 404
        
        from flask import send_file
        return send_file(
            dataset_path,
            as_attachment=True,
            download_name='bonetox-new.csv',
            mimetype='text/csv'
        )
    except Exception as e:
        print(f"下载数据集失败: {e}")
        return f"下载失败: {str(e)}", 500

# @app.route('/download_pdf')
# def download_pdf():
#     # 这里可以实现PDF报告生成
#     return "PDF下载功能待实现", 501

if __name__ == '__main__':
    # 启动时清理旧图片
    cleanup_old_images()
    print(f"临时图片目录: {TMP_IMG_DIR}")
    app.run(debug=True)
