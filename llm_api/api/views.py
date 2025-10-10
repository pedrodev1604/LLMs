import ast
import json
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.http import JsonResponse

import pandas as pd
import re


from api.llm import calcular_metricas, extrair_dados, extrair_dados_rule, extrair_entidades_somente_llm, extrair_entidades_somente_llm_rule, gerar_relatorio_entidades_somente_llm, gerar_relatorio_entidades_somente_llm_rule, gerar_relatorio_metricas_pdf, gerar_relatorio_pdf, gerar_relatorio_rules_metricas_pdf, gerar_relatorio_rules_pdf


def parse_labels(value):
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            return value[0]
        return value
    if isinstance(value, str):
        try:
            value = value.strip()

            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]

            value = re.sub(r'\s+', ' ', value)

            parsed = ast.literal_eval(value)

            if isinstance(parsed, tuple) and len(parsed) == 1 and isinstance(parsed[0], list):
                return parsed[0]

            return parsed
        except Exception as e:
            print(f"Erro ao parsear labels: {e}")
            print(f"Valor recebido: {value}")
            return None
    return None


import pandas as pd

def gerar_tabela_comparativa(tokens_labels, tokens_pred):
    tokens = [t for t, _ in tokens_labels]
    labels_csv = [l for _, l in tokens_labels]
    labels_llm = [l for _, l in tokens_pred]

    df = pd.DataFrame({
        'Token': tokens,
        'CRF': labels_csv,
        'gpt-4.1-nano': labels_llm
    })

    return df

def gerar_tabela_comparativa_rule(tokens_labels, tokens_pred):
    tokens = [t for t, _ in tokens_labels]
    labels_csv = [l for _, l in tokens_labels]
    labels_llm = [l for _, l in tokens_pred]

    df = pd.DataFrame({
        'Token': tokens,
        'BOSQUE': labels_csv,
        'gpt-4.1-nano': labels_llm
    })

    return df

def limpar_y(y):
    """
    Corrige estrutura de y_true ou y_pred removendo listas aninhadas.
    """
    novo_y = []
    for seq in y:
        nova_seq = []
        if isinstance(seq, list):
            for item in seq:
                if isinstance(item, list):
                    if len(item) >= 2 and isinstance(item[-1], str):
                        nova_seq.append(item[-1])  
                    else:
                        nova_seq.append('O') 
                elif isinstance(item, str):
                    nova_seq.append(item)
                else:
                    nova_seq.append(str(item))
            novo_y.append(nova_seq)
    return novo_y


class ExtrairCSVAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'Arquivo CSV não enviado.'},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            df = pd.read_csv(file)

            if 'sentence' not in df.columns or 'labels' not in df.columns:
                return Response({'error': 'O CSV precisa ter as colunas "sentence" e "labels".'},
                                status=status.HTTP_400_BAD_REQUEST)

            comparacoes = []
            erros = []

            y_true = []
            y_pred = []
            tokens_por_sentenca = []
            sentences = []

            for _, row in df.iterrows():
                frase = row['sentence']
                labels = parse_labels(row['labels'])

                if labels is None:
                    erros.append({'sentence': frase, 'erro': 'Labels inválidos'})
                    continue

                try:
                    dados = extrair_dados(frase)

                    if not dados or 'result' not in dados:
                        erros.append({'sentence': frase, 'erro': 'Erro na resposta da LLM'})
                        continue

                    resultado_llm = dados['result']

                    tokens_labels = [(t[0], t[1]) for t in labels]
                    tokens_pred = [(t[0], t[1]) for t in resultado_llm]

                    min_len = min(len(tokens_labels), len(tokens_pred))
                    tokens_labels = tokens_labels[:min_len]
                    tokens_pred = tokens_pred[:min_len]

                    df_comp = gerar_tabela_comparativa(tokens_labels, tokens_pred)
                    comparacoes.append({'sentence': frase, 'tabela': df_comp})

                    y_true.append([label for _, label in tokens_labels])
                    y_pred.append([label for _, label in tokens_pred])
                    tokens_por_sentenca.append([token for token, _ in tokens_labels])
                    sentences.append(frase)

                except Exception as e:
                    erros.append({'sentence': frase, 'erro': str(e)})
                    continue

            pdf_path = gerar_relatorio_pdf(comparacoes)

            y_true = limpar_y(y_true)
            y_pred = limpar_y(y_pred)

            metricas = calcular_metricas(y_true, y_pred)
            metricas_pdf_path = gerar_relatorio_metricas_pdf(metricas, filename="relatorio_metricas.pdf")

            entidades_extra_llm = extrair_entidades_somente_llm(
                y_true, y_pred, tokens_por_sentenca, sentences
            )
            entidades_pdf_path = gerar_relatorio_entidades_somente_llm(
                entidades_extra_llm, filename="relatorio_entidades_somente_llm.pdf"
            )

            return JsonResponse({
                'message': 'Relatórios gerados com sucesso!',
                'pdf_comparacao': pdf_path,
                'pdf_metricas': metricas_pdf_path,
                'pdf_entidades_extra_llm': entidades_pdf_path,
                'erros': erros
            }, status=200)

        except Exception as e:
            return Response({'error': str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)


from conllu import parse_incr
import io

class ExtrairConlluAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'Arquivo CONLLU não enviado.'}, status=400)

        try:
            content = file.read().decode('utf-8')
            sentences_parsed = list(parse_incr(io.StringIO(content)))
        except Exception as e:
            return Response({'error': str(e)}, status=400)

        try:
            comparacoes = []
            erros = []
            y_true = []
            y_pred = []
            tokens_por_sentenca = []
            sentences = []

            for sentence in sentences_parsed:
                tokens_labels = []
                frase = " ".join([token["form"] for token in sentence])

                for token in sentence:
                    if "form" in token and "upos" in token:
                        tokens_labels.append((token["form"], token["upos"]))

                try:
                    dados = extrair_dados_rule(frase)
                    if not dados or 'result' not in dados:
                        erros.append({'sentence': frase, 'erro': 'Erro na resposta da LLM'})
                        continue

                    resultado_llm = dados['result']
                    tokens_pred = [(t[0], t[1]) for t in resultado_llm]

                    min_len = min(len(tokens_labels), len(tokens_pred))
                    tokens_labels = tokens_labels[:min_len]
                    tokens_pred = tokens_pred[:min_len]

                    df_comp = gerar_tabela_comparativa_rule(tokens_labels, tokens_pred)
                    comparacoes.append({'sentence': frase, 'tabela': df_comp})

                    y_true.append([label for _, label in tokens_labels])
                    y_pred.append([label for _, label in tokens_pred])
                    tokens_por_sentenca.append([token for token, _ in tokens_labels])
                    sentences.append(frase)

                except Exception as e:
                    erros.append({'sentence': frase, 'erro': str(e)})
                    continue

            y_true = limpar_y(y_true)
            y_pred = limpar_y(y_pred)

            metricas = calcular_metricas(y_true, y_pred)
            print("\n--- MÉTRICAS DO MODELO BASEADO EM REGRAS ---")
            for metrica in metricas:
                print(f"{metrica['type']}: Precision = {metrica['precision']}, Recall = {metrica['recall']}")
            print("--------------------------------------------\n")
            pdf_path = gerar_relatorio_rules_pdf(comparacoes)
            metricas_pdf_path = gerar_relatorio_rules_metricas_pdf(metricas, filename="relatorio_metricas_rules.pdf")

            entidades_extra_llm = extrair_entidades_somente_llm_rule(
                y_true, y_pred, tokens_por_sentenca, sentences
            )
            entidades_pdf_path = gerar_relatorio_entidades_somente_llm_rule(
                entidades_extra_llm, filename="relatorio_entidades_somente_llm.pdf"
            )

            return JsonResponse({
                'message': 'Relatórios gerados com sucesso!',
                'pdf_comparacao': pdf_path,
                'pdf_metricas': metricas_pdf_path,
                'pdf_entidades_extra_llm': entidades_pdf_path,
                'erros': erros
            }, status=200)

        except Exception as e:
            return Response({'error': str(e)}, status=500)
        

