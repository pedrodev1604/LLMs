import json
import os
import re
from langchain_openai import ChatOpenAI
from api.prompt import get_prompt, get_prompt_rule
from config import settings
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

llm = ChatOpenAI(
    model='gpt-4.1-nano',
    openai_api_key=''    
)

def extrair_dados(frase):
    prompt = get_prompt(frase)
    resposta = llm.invoke(prompt)
    try:
        return json.loads(resposta.content)
    except json.JSONDecodeError:
        try:
            content_str = resposta.content.strip()
            if content_str.startswith('"') and content_str.endswith('"'):
                content_str = content_str[1:-1]
            content_str = content_str.encode('utf-8').decode('unicode_escape')
            return json.loads(content_str)
        except Exception as e:
            print(f"Erro ao limpar e carregar JSON: {e}")
            return None

# ✅ Função auxiliar
def safe_str(val):
    if isinstance(val, list):
        return ', '.join(map(str, val))
    elif val is None:
        return ''
    else:
        return str(val)

def limpar_reparar_json(resposta):
    try:
        json_raw = re.search(r'\{.*\}', resposta, re.DOTALL).group()
        json_raw = json_raw.replace("'", '"')
        json_raw = re.sub(r'\]\s*\[', '], [', json_raw)
        json_raw = json_raw.replace('\n', ' ').strip()
        json_raw = re.sub(r'\s{2,}', ' ', json_raw)
        return json.loads(json_raw)
    except Exception as e:
        print(f"Erro ao limpar e carregar JSON: {e}")
        print(f"JSON bruto: {json_raw[:500]}...")
        return None



def calcular_metricas(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)

    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    micro_precision = report["micro avg"]["precision"]
    micro_recall = report["micro avg"]["recall"]

    return [
        {'type': 'Macro Average', 'precision': round(macro_precision, 2), 'recall': round(macro_recall, 2)},
        {'type': 'Micro Average', 'precision': round(micro_precision, 2), 'recall': round(micro_recall, 2)},
    ]


def gerar_relatorio_pdf(comparacoes, filename="relatorio_comparativo.pdf"):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    media_root = settings.MEDIA_ROOT
    os.makedirs(media_root, exist_ok=True)
    file_path = os.path.join(media_root, filename)

    doc = SimpleDocTemplate(file_path, pagesize=A4,
                             rightMargin=40, leftMargin=40,
                             topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    for idx, comp in enumerate(comparacoes, start=1):
        sentence = comp['sentence']
        df = comp['tabela']

        # ✅ Garantir que sentence é string
        title = f"Sentence {idx}: '{safe_str(sentence)}'"
        story.append(Paragraph(title, styles['Heading3']))
        story.append(Spacer(1, 10))

        # Dados da tabela
        data = [['Token', 'CRF', 'gpt-4.1-nano']]
        for _, row in df.iterrows():
            data.append([
                safe_str(row['Token']),
                safe_str(row['CRF']),
                safe_str(row['gpt-4.1-nano'])
            ])

        table = Table(data, colWidths=[180, 100, 100])

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ])
        table.setStyle(style)

        story.append(table)
        story.append(Spacer(1, 20))

    doc.build(story)
    return file_path

def gerar_relatorio_metricas_pdf(metricas, filename="relatorio_metricas.pdf"):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    media_root = settings.MEDIA_ROOT
    os.makedirs(media_root, exist_ok=True)
    file_path = os.path.join(media_root, filename)

    doc = SimpleDocTemplate(file_path, pagesize=A4,
                             rightMargin=40, leftMargin=40,
                             topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("NER Metrics Report: CRF vs GPT-4.1-Nano", styles['Heading2']))
    story.append(Spacer(1, 20))

    data = [['Tipo', 'Precision', 'Recall']]
    for metrica in metricas:
        data.append([
            safe_str(metrica['type']),
            f"{metrica['precision']:.2f}",
            f"{metrica['recall']:.2f}"
        ])

    table = Table(data, colWidths=[120, 80, 80])

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ])
    table.setStyle(style)

    story.append(table)
    doc.build(story)
    return file_path




from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def gerar_relatorio_entidades_somente_llm(dados, filename="relatorio_entidades_somente_llm.pdf"):
    media_root = settings.MEDIA_ROOT
    os.makedirs(media_root, exist_ok=True)
    file_path = os.path.join(media_root, filename)

    doc = SimpleDocTemplate(file_path, pagesize=A4,
                             rightMargin=40, leftMargin=40,
                             topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    # Estilo para as sentenças longas
    sentence_style = ParagraphStyle(
        name="SentenceStyle",
        parent=styles['Normal'],
        fontSize=7,
        leading=9
    )

    story = []

    title = "Report: Entities Detected Only by LLM"
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 20))

    if not dados:
        story.append(Paragraph("Nenhuma entidade extra detectada pela LLM.", styles['Normal']))
    else:
        data = [['Token', 'gpt-4.1-nano', 'CRF', 'Sentence']]

        for item in dados:
            sentence_para = Paragraph(str(item.get('Sentence', '')), sentence_style)
            data.append([
                str(item.get('Token', '')),
                str(item.get('gpt-4.1-nano', '')),
                str(item.get('CRF', '')),
                sentence_para
            ])

        table = Table(data, colWidths=[70, 70, 70, 280]) 

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-2, -1), 'CENTER'),
            ('ALIGN', (-1, 0), (-1, -1), 'LEFT'), 
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ])
        table.setStyle(style)

        story.append(table)

    doc.build(story)
    return file_path



def extrair_entidades_somente_llm(y_true, y_pred, tokens_por_sentenca, sentences):
    """
    Retorna uma lista de dicionários com tokens que a LLM rotulou como entidade
    mas o CRF rotulou como 'O'.
    
    y_true, y_pred: listas de listas de labels (ex.: [['O', 'B-per', ...], ...])
    tokens_por_sentenca: lista de listas de tokens (ex.: [['John', 'lives', ...], ...])
    sentences: lista de sentenças completas como string
    """
    resultados = []

    for tokens, true_labels, pred_labels, sent in zip(tokens_por_sentenca, y_true, y_pred, sentences):
        for token, true_label, pred_label in zip(tokens, true_labels, pred_labels):
            if true_label == 'O' and pred_label != 'O':
                resultados.append({
                    'Token': token,
                    'gpt-4.1-nano': pred_label,
                    'CRF': true_label,
                    'Sentence': sent
                })
    return resultados


def extrair_dados_rule(frase):
    prompt = get_prompt_rule(frase)
    resposta = llm.invoke(prompt)
    try:
        return json.loads(resposta.content)
    except json.JSONDecodeError:
        try:
            content_str = resposta.content.strip()
            if content_str.startswith('"') and content_str.endswith('"'):
                content_str = content_str[1:-1]
            content_str = content_str.encode('utf-8').decode('unicode_escape')
            return json.loads(content_str)
        except Exception as e:
            print(f"Erro ao limpar e carregar JSON: {e}")
            return None
        

def gerar_relatorio_rules_pdf(comparacoes, filename="relatorio_comparativo_rules.pdf"):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    media_root = settings.MEDIA_ROOT
    os.makedirs(media_root, exist_ok=True)
    file_path = os.path.join(media_root, filename)

    doc = SimpleDocTemplate(file_path, pagesize=A4,
                             rightMargin=40, leftMargin=40,
                             topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    for idx, comp in enumerate(comparacoes, start=1):
        sentence = comp['sentence']
        df = comp['tabela']

        # ✅ Garantir que sentence é string
        title = f"Sentence {idx}: '{safe_str(sentence)}'"
        story.append(Paragraph(title, styles['Heading3']))
        story.append(Spacer(1, 10))

        # Dados da tabela
        data = [['Token', 'BOSQUE', 'gpt-4.1-nano']]
        for _, row in df.iterrows():
            data.append([
                safe_str(row['Token']),
                safe_str(row['BOSQUE']),
                safe_str(row['gpt-4.1-nano'])
            ])

        table = Table(data, colWidths=[180, 100, 100])

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ])
        table.setStyle(style)

        story.append(table)
        story.append(Spacer(1, 20))

    doc.build(story)
    return file_path


def gerar_relatorio_rules_metricas_pdf(metricas, filename="relatorio_metricas_rules.pdf"):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    media_root = settings.MEDIA_ROOT
    os.makedirs(media_root, exist_ok=True)
    file_path = os.path.join(media_root, filename)

    doc = SimpleDocTemplate(file_path, pagesize=A4,
                             rightMargin=40, leftMargin=40,
                             topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("POS Tagging Metrics Report: BOSQUE vs GPT-4.1-Nano", styles['Heading2']))
    story.append(Spacer(1, 20))

    data = [['Tipo', 'Precision', 'Recall']]
    for metrica in metricas:
        data.append([
            safe_str(metrica['type']),
            f"{metrica['precision']:.2f}",
            f"{metrica['recall']:.2f}"
        ])

    table = Table(data, colWidths=[120, 80, 80])

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ])
    table.setStyle(style)

    story.append(table)
    doc.build(story)
    return file_path



def gerar_relatorio_entidades_somente_llm_rule(dados, filename="relatorio_entidades_somente_llm_rule.pdf"):
    media_root = settings.MEDIA_ROOT
    os.makedirs(media_root, exist_ok=True)
    file_path = os.path.join(media_root, filename)

    doc = SimpleDocTemplate(file_path, pagesize=A4,
                             rightMargin=40, leftMargin=40,
                             topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    # Estilo para as sentenças longas
    sentence_style = ParagraphStyle(
        name="SentenceStyle",
        parent=styles['Normal'],
        fontSize=7,
        leading=9
    )

    story = []

    title = "Report: Entities Detected Only by LLM"
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 20))

    if not dados:
        story.append(Paragraph("Nenhuma entidade extra detectada pela LLM.", styles['Normal']))
    else:
        data = [['Token', 'gpt-4.1-nano', 'BOSQUE', 'Sentence']]

        for item in dados:
            sentence_para = Paragraph(str(item.get('Sentence', '')), sentence_style)
            data.append([
                str(item.get('Token', '')),
                str(item.get('gpt-4.1-nano', '')),
                str(item.get('BOSQUE', '')),
                sentence_para
            ])

        table = Table(data, colWidths=[70, 70, 70, 280]) 

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-2, -1), 'CENTER'),
            ('ALIGN', (-1, 0), (-1, -1), 'LEFT'), 
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ])
        table.setStyle(style)

        story.append(table)

    doc.build(story)
    return file_path

def extrair_entidades_somente_llm_rule(y_true, y_pred, tokens_por_sentenca, sentences):
    resultados = []

    for tokens, true_labels, pred_labels, sent in zip(tokens_por_sentenca, y_true, y_pred, sentences):
        for token, true_label, pred_label in zip(tokens, true_labels, pred_labels):
            # Considera que no BOSQUE "sem entidade" é "", "_", ou "O"
            bosqueless = true_label in ("", "_", "O")  
            
            # Se no BOSQUE não tem entidade mas na predição da LLM tem
            if bosqueless and pred_label != '_':
                resultados.append({
                    'Token': token,
                    'gpt-4.1-nano': pred_label,
                    'BOSQUE': true_label,
                    'Sentence': sent
                })
    return resultados


def extrair_dados_frase(frase):
    prompt = get_prompt_rule(frase)
    resposta = llm.invoke(prompt)
    try:
        return json.loads(resposta.content)
    except json.JSONDecodeError:
        try:
            content_str = resposta.content.strip()
            if content_str.startswith('"') and content_str.endswith('"'):
                content_str = content_str[1:-1]
            content_str = content_str.encode('utf-8').decode('unicode_escape')
            return json.loads(content_str)
        except Exception as e:
            print(f"Erro ao limpar e carregar JSON: {e}")
            return None