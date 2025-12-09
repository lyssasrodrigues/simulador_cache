#Codigo do desenvolvimento das atividades 2 e 4

# Importações necessárias
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import dearpygui.dearpygui as dpg
import time, sys, os
from datetime import datetime
import math
import threading 
import platform 

# ------------------------------------------------------------------------------
# 1. CLASSES E VARIÁVEIS GLOBAIS
# ------------------------------------------------------------------------------

# Redirecionador de saída (Faz o print ir para a caixa de texto)
class DPGRedirector:
    def __init__(self, tag):
        self.tag = tag
    def write(self, text):
        try:
            if dpg.does_item_exist(self.tag):
                current = dpg.get_value(self.tag) or ""
                if len(current) > 10000: current = current[-5000:] # Buffer maior
                dpg.set_value(self.tag, current + text)
                # Scroll automático para o fim
                dpg.set_y_scroll(self.tag, -1.0)
        except: pass
    def flush(self): pass

plot_series_tags = []
algoritmo_escolhido = "FIFO"
resultados = []

# Estado da Animação
anim_state = {
    'running': False,
    'speed': 0.1,
    'last_update': 0,
    'padrao': [],
    'index': 0,
    'wt_ram_writes': 0,
    'wb_ram_writes': 0,
    'wb_cache': {}, 
    'wt_cache': set(),
    'cache_size_blocks': 4,
    'msg_wt': "",
    'msg_wb': ""
}

# ------------------------------------------------------------------------------
# 2. FUNÇÕES AUXILIARES
# ------------------------------------------------------------------------------

def play_sound_async(type_sound):
    def _sound_worker():
        try:
            if platform.system() == "Windows":
                import winsound
                if type_sound == 'WT': winsound.Beep(2000, 30)
                elif type_sound == 'WB': winsound.Beep(400, 150)
        except: pass
    threading.Thread(target=_sound_worker, daemon=True).start()

def is_power_of_two(x):
    return (x != 0) and ((x & (x - 1)) == 0)

def gerar_padrao_realista(acessos, memory_size, regioes_quentes, prob_temporal, prob_espacial, prob_quente, bloco_tamanho):
    padrao = []
    endereco_anterior = random.randint(0, memory_size - 1)
    max_addr = memory_size - 1
    for _ in range(acessos):
        r = random.random()
        if r < prob_temporal: endereco = endereco_anterior
        elif r < prob_temporal + prob_espacial:
            endereco = max(0, min(max_addr, endereco_anterior + random.randint(-16, 16)))
        elif r < prob_temporal + prob_espacial + prob_quente:
            endereco = min(max_addr, random.choice(regioes_quentes) + random.randint(0, 3))
        else: endereco = random.randint(0, max_addr)
        padrao.append(endereco)
        endereco_anterior = endereco
    return padrao

def gerar_padrao_com_rw(acessos, memory_size, prob_write=0.3):
    regioes = [64, 1024, 8192, 32000]
    raw_padrao = gerar_padrao_realista(acessos, memory_size, regioes, 0.3, 0.3, 0.3, 1)
    padrao_final = []
    for addr in raw_padrao:
        is_write = random.random() < prob_write
        padrao_final.append((addr, is_write))
    return padrao_final

# ------------------------------------------------------------------------------
# 3. LÓGICA DE SIMULAÇÃO DE CACHE (CORE)
# ------------------------------------------------------------------------------

def simular_cache_FIFO(padrao, cache_lines, assoc, b_size):
    sets = max(1, cache_lines // assoc)
    cache = [[] for _ in range(sets)]
    hit_log = []
    for addr in padrao:
        blk = addr // b_size
        idx = blk % sets
        st = cache[idx]
        if blk in st: hit_log.append(1)
        else:
            hit_log.append(0)
            if len(st) < assoc: st.append(blk)
            else: 
                st.pop(0)
                st.append(blk)
    return [], hit_log

def simular_cache_LRU(padrao, cache_lines, assoc, b_size):
    sets = max(1, cache_lines // assoc)
    cache = [deque() for _ in range(sets)]
    hit_log = []
    for addr in padrao:
        blk = addr // b_size
        idx = blk % sets
        st = cache[idx]
        if blk in st:
            hit_log.append(1)
            st.remove(blk)
            st.append(blk)
        else:
            hit_log.append(0)
            if len(st) >= assoc: st.popleft()
            st.append(blk)
    return [], hit_log

def simular_cache_LFU(padrao, cache_lines, assoc, b_size):
    sets = max(1, cache_lines // assoc)
    cache = [{} for _ in range(sets)]
    hit_log = []
    for addr in padrao:
        blk = addr // b_size
        idx = blk % sets
        st = cache[idx]
        if blk in st:
            hit_log.append(1)
            st[blk] += 1
        else:
            hit_log.append(0)
            if len(st) < assoc: st[blk] = 1
            else:
                rem = min(st, key=st.get)
                del st[rem]
                st[blk] = 1
    return [], hit_log

def simular_cache_RANDOM(padrao, cache_lines, assoc, b_size):
    sets = max(1, cache_lines // assoc)
    cache = [[] for _ in range(sets)]
    hit_log = []
    for addr in padrao:
        blk = addr // b_size
        idx = blk % sets
        st = cache[idx]
        if blk in st: hit_log.append(1)
        else:
            hit_log.append(0)
            if len(st) < assoc: st.append(blk)
            else: st[random.randint(0, assoc-1)] = blk
    return [], hit_log

def simulacao_monte_carlo(n_sim, acessos, mem, lines, assoc, reg, probs, b_size, algo):
    res_taxas = []
    
    # Executa N simulações
    for i in range(n_sim):
        p = gerar_padrao_realista(acessos, mem, reg, *probs, b_size)
        if algo=='FIFO': _, l = simular_cache_FIFO(p, lines, assoc, b_size)
        elif algo=='LRU': _, l = simular_cache_LRU(p, lines, assoc, b_size)
        elif algo=='LFU': _, l = simular_cache_LFU(p, lines, assoc, b_size)
        else: _, l = simular_cache_RANDOM(p, lines, assoc, b_size)
        
        taxa = sum(l)/len(l) if l else 0
        res_taxas.append(taxa)

    media = np.mean(res_taxas)
    desvio = np.std(res_taxas)
    maximo = max(res_taxas)
    minimo = min(res_taxas)

    print(f"--- Resultados: {acessos} Acessos - Bloco de {b_size} ---")
    print(f"Média da Taxa de Acerto: {media:.4f}")
    print(f"Desvio Padrão: {desvio:.4f}")
    print(f"Máximo: {maximo:.4f}, Mínimo: {minimo:.4f}")
    print("-" * 30 + "\n")

    return media

# Simulação Multinível
def simular_cache_multinivel_core(padrao, niveis_config, algoritmos):
    n_niveis = len(niveis_config)
    hits_por_nivel = [0] * n_niveis
    misses_por_nivel = [0] * n_niveis
    caches = []
    for i in range(n_niveis):
        cache_lines, associatividade, bloco_tamanho = niveis_config[i]
        num_conjuntos = max(1, cache_lines // associatividade)
        if algoritmos[i] == 'LFU': cache = [{} for _ in range(num_conjuntos)]
        elif algoritmos[i] == 'LRU': cache = [deque() for _ in range(num_conjuntos)]
        else: cache = [[] for _ in range(num_conjuntos)]
        caches.append((cache, num_conjuntos, associatividade, bloco_tamanho, algoritmos[i]))
    
    for endereco in padrao:
        dados_encontrados = False
        for nivel in range(n_niveis):
            cache, num_conjuntos, associatividade, bloco_tamanho, algoritmo = caches[nivel]
            bloco = endereco // bloco_tamanho
            conjunto = bloco % num_conjuntos
            conjunto_atual = cache[conjunto]
            is_hit = False
            if algoritmo == 'LFU':
                if bloco in conjunto_atual:
                    conjunto_atual[bloco] += 1
                    is_hit = True
            elif algoritmo == 'LRU':
                if bloco in conjunto_atual:
                    conjunto_atual.remove(bloco)
                    conjunto_atual.append(bloco)
                    is_hit = True
            else: 
                if bloco in conjunto_atual: is_hit = True
            if is_hit:
                hits_por_nivel[nivel] += 1
                dados_encontrados = True
                break
            else:
                misses_por_nivel[nivel] += 1
        
        if not dados_encontrados:
            for nivel in range(n_niveis):
                cache, num_conjuntos, associatividade, bloco_tamanho, algoritmo = caches[nivel]
                bloco = endereco // bloco_tamanho
                conjunto = bloco % num_conjuntos
                conjunto_atual = cache[conjunto]
                if algoritmo == 'LFU':
                    if len(conjunto_atual) < associatividade: conjunto_atual[bloco] = 1
                    else:
                        del conjunto_atual[min(conjunto_atual, key=conjunto_atual.get)]
                        conjunto_atual[bloco] = 1
                elif algoritmo == 'LRU':
                    if len(conjunto_atual) < associatividade: conjunto_atual.append(bloco)
                    else:
                        conjunto_atual.popleft()
                        conjunto_atual.append(bloco)
                elif algoritmo == 'FIFO':
                    if len(conjunto_atual) < associatividade: conjunto_atual.append(bloco)
                    else:
                        conjunto_atual.pop(0)
                        conjunto_atual.append(bloco)
                else: 
                    if len(conjunto_atual) < associatividade: conjunto_atual.append(bloco)
                    else:
                        conjunto_atual[random.randint(0, associatividade - 1)] = bloco
    hit_rates = []
    total_acessos = len(padrao)
    if total_acessos > 0:
        hit_rates.append(hits_por_nivel[0] / total_acessos)
        for i in range(1, n_niveis):
            acessos_nivel = misses_por_nivel[i-1]
            hit_rates.append(hits_por_nivel[i] / acessos_nivel if acessos_nivel > 0 else 0.0)
    else:
        hit_rates = [0.0] * n_niveis
    return hit_rates

def calcular_tempo_medio_acesso(hit_rates, tempos_acesso):
    n_niveis = len(hit_rates)
    t_nivel_seguinte = tempos_acesso[-1]
    for i in range(n_niveis - 1, -1, -1):
        hit_rate = hit_rates[i]
        t_nivel_atual = tempos_acesso[i]
        t_nivel_seguinte = hit_rate * t_nivel_atual + (1 - hit_rate) * t_nivel_seguinte
    return t_nivel_seguinte

def simulacao_monte_carlo_multinivel_wrapper(n_sim, acessos, mem, niveis, algos, tempos, reg, probs, b_size):
    hit_rates_total = [[] for _ in range(len(niveis))]
    tempos_medios = []
    for i in range(n_sim):
        padrao = gerar_padrao_realista(acessos, mem, reg, *probs, b_size)
        hit_rates = simular_cache_multinivel_core(padrao, niveis, algos)
        for nivel, taxa in enumerate(hit_rates):
            hit_rates_total[nivel].append(taxa)
        tempos_medios.append(calcular_tempo_medio_acesso(hit_rates, tempos))
    return [np.mean(taxas) for taxas in hit_rates_total], np.mean(tempos_medios)

# ------------------------------------------------------------------------------
# 4. LÓGICA DA ANIMAÇÃO
# ------------------------------------------------------------------------------

def reset_animacao_state():
    global anim_state
    anim_state['padrao'] = gerar_padrao_com_rw(100, 256, prob_write=0.4) 
    anim_state['index'] = 0
    anim_state['running'] = False
    anim_state['wt_ram_writes'] = 0
    anim_state['wb_ram_writes'] = 0
    anim_state['wb_cache'] = {}
    anim_state['wt_cache'] = {}
    
    if dpg.does_item_exist("anim_progress"):
        dpg.set_value("anim_progress", 0.0)
        dpg.set_value("wt_counter", "0")
        dpg.set_value("wb_counter", "0")
        dpg.set_value("wt_status", "Pronto")
        dpg.set_value("wb_status", "Pronto")
        dpg.configure_item("wt_bus_line", color=(100, 100, 100, 255), thickness=2)
        dpg.configure_item("wb_bus_line", color=(100, 100, 100, 255), thickness=2)

def update_animation():
    global anim_state
    if not anim_state['running']: return
    if time.time() - anim_state['last_update'] < anim_state['speed']: return
    anim_state['last_update'] = time.time()
    
    idx = anim_state['index']
    if idx >= len(anim_state['padrao']):
        anim_state['running'] = False
        dpg.set_value("wt_status", "Concluído!")
        dpg.set_value("wb_status", "Concluído!")
        return

    addr, is_write = anim_state['padrao'][idx]
    op_str = "ESCRITA (W)" if is_write else "LEITURA (R)"
    
    num_blocks = 4
    tag = addr // num_blocks
    cache_idx = addr % num_blocks
    
    # --- Write-Through ---
    wt_hit = False
    wt_msg = f"CPU {op_str} Endereço {addr} -> "
    
    if not isinstance(anim_state['wt_cache'], dict): anim_state['wt_cache'] = {}
    
    if cache_idx in anim_state['wt_cache'] and anim_state['wt_cache'][cache_idx] == tag:
        wt_hit = True
        wt_msg += "HIT na Cache. "
    else:
        wt_msg += "MISS. Trazendo da RAM. "
        anim_state['wt_cache'][cache_idx] = tag
    
    bus_active_wt = False
    if is_write:
        wt_msg += "Escrevendo na RAM imediatamente!"
        anim_state['wt_ram_writes'] += 1
        bus_active_wt = True
        play_sound_async('WT')
    else:
        if not wt_hit: bus_active_wt = True 
    
    # --- Write-Back ---
    wb_hit = False
    wb_msg = f"CPU {op_str} Endereço {addr} -> "
    bus_active_wb = False
    current_block = anim_state['wb_cache'].get(cache_idx)
    
    if current_block and current_block['tag'] == tag:
        wb_hit = True
        wb_msg += "HIT. "
        if is_write:
            current_block['dirty'] = True
            wb_msg += "Marcando bloco como SUJO."
    else:
        wb_msg += "MISS. "
        if current_block and current_block['dirty']:
            wb_msg += "Expulsando bloco SUJO -> RAM! "
            anim_state['wb_ram_writes'] += 1
            bus_active_wb = True
            play_sound_async('WB')
        anim_state['wb_cache'][cache_idx] = {'tag': tag, 'dirty': is_write}
        if is_write: wb_msg += "Novo bloco marcado como SUJO."

    dpg.set_value("wt_status", wt_msg)
    dpg.set_value("wb_status", wb_msg)
    dpg.set_value("wt_counter", str(anim_state['wt_ram_writes']))
    dpg.set_value("wb_counter", str(anim_state['wb_ram_writes']))
    dpg.set_value("anim_progress", (idx + 1) / len(anim_state['padrao']))
    
    if bus_active_wt: dpg.configure_item("wt_bus_line", color=(255, 255, 0, 255), thickness=6)
    else: dpg.configure_item("wt_bus_line", color=(50, 50, 50, 255), thickness=2)
        
    if bus_active_wb: dpg.configure_item("wb_bus_line", color=(255, 50, 50, 255), thickness=6)
    else: dpg.configure_item("wb_bus_line", color=(50, 50, 50, 255), thickness=2)

    anim_state['index'] += 1

# ------------------------------------------------------------------------------
# 5. CALLBACKS (TODOS)
# ------------------------------------------------------------------------------

def start_anim_callback(sender, app_data):
    if anim_state['index'] >= len(anim_state['padrao']): reset_animacao_state()
    anim_state['running'] = True

def pause_anim_callback(sender, app_data): 
    anim_state['running'] = False

def reset_anim_callback(sender, app_data): 
    reset_animacao_state()

def update_speed_callback(sender, app_data):
    val = dpg.get_value(sender)
    anim_state['speed'] = 1.0 / (val if val > 0 else 1)

def selecionar_algoritmo(sender, app_data):
    global algoritmo_escolhido
    algoritmo_escolhido = app_data

def limpar_plots(sender, app_data):
    global plot_series_tags
    for t in plot_series_tags:
        if dpg.does_item_exist(t): dpg.delete_item(t)
    plot_series_tags.clear()
    dpg.set_value("barra", 0.0)
    dpg.set_value("Resumo", "") # Limpa também o texto

def limpar_ultimo_plot(sender, app_data):
    global plot_series_tags
    if plot_series_tags:
        t = plot_series_tags.pop()
        if dpg.does_item_exist(t): dpg.delete_item(t)

def limpar_heatmap(sender, app_data):
    if dpg.does_item_exist("heatmap_image"): dpg.delete_item("heatmap_image")
    if dpg.does_item_exist("heatmap_text"): dpg.delete_item("heatmap_text")

def close_callback(sender, app_data):
    dpg.stop_dearpygui()

def mapa_temporal_blocos_global(sender, app_data):
    try:
        mem = int(dpg.get_value("memory_size"))
        bloco = int(dpg.get_value("blocos").split(",")[0])
        acessos = int(dpg.get_value("acessos"))
        
        regioes = [64, 1024, 8192, 32768]
        padrao = gerar_padrao_realista(acessos, mem, regioes, 0.3, 0.3, 0.4, bloco)
        
        res_temp = 100
        num_janelas = max(1, len(padrao) // res_temp)
        res_blocos = 256
        num_blocos = max(1, mem // bloco)
        num_blocos_agrup = max(1, num_blocos // res_blocos)
        
        heatmap = np.zeros((num_blocos_agrup, num_janelas), dtype=int)
        
        for i, addr in enumerate(padrao):
            t = min(i // res_temp, num_janelas - 1)
            b = addr // bloco
            b_agrup = min(b // res_blocos, num_blocos_agrup - 1)
            heatmap[b_agrup][t] += 1
            
        plt.figure(figsize=(6, 4))
        plt.imshow(heatmap, cmap='hot', aspect='auto', origin='lower', interpolation='nearest')
        plt.colorbar(label="Acessos")
        plt.title(f"Heatmap (Bloco: {bloco}B)")
        plt.xlabel("Tempo")
        plt.ylabel("Bloco Memória")
        plt.tight_layout()
        
        if not os.path.exists('Heatmaps'): os.makedirs('Heatmaps')
        fn = f'Heatmaps/heatmap_{datetime.now().strftime("%H%M%S")}.png'
        plt.savefig(fn, dpi=100, bbox_inches='tight')
        plt.close()
        
        if dpg.does_item_exist("heatmap_texture"): dpg.delete_item("heatmap_texture")
        w, h, c, data = dpg.load_image(fn)
        with dpg.texture_registry():
            dpg.add_static_texture(w, h, data, tag="heatmap_texture")
            
        if dpg.does_item_exist("heatmap_image"):
            dpg.delete_item("heatmap_image")
            dpg.delete_item("heatmap_text")
            
        dpg.add_image("heatmap_texture", parent="heatmap_group", tag="heatmap_image", width=540, height=450)
        dpg.add_text(f"Visualizando Bloco: {bloco} bytes", parent="heatmap_group", tag="heatmap_text")
        
    except Exception as e:
        print(f"Erro heatmap: {str(e)}")

def rodar_simulacao_aba1(sender, app_data):
    try:
        mem = dpg.get_value("memory_size")
        acs = dpg.get_value("acessos")
        c_size = dpg.get_value("tamanho_cache")
        assoc = dpg.get_value("associatividade")
        n_sim = dpg.get_value("n_simulacoes")
        blocos = [int(x) for x in dpg.get_value("blocos").split(",")]
        probs = (dpg.get_value("prob_temporal"), dpg.get_value("prob_espacial"), dpg.get_value("prob_quente"))
        algo = dpg.get_value("combo_algoritmo")
        reg = [64, 1024, 8192, 32000]

        res_aba1 = []
        dpg.set_value("barra", 0.0)
        
        start_t = time.time()
        print(f"\n--- Iniciando Simulação: {algo} ---") # LOG INICIAL
        
        for i, b in enumerate(blocos):
            lines = c_size // b
            taxa = simulacao_monte_carlo(n_sim, acs, mem, lines, assoc, reg, probs, b, algo)
            res_aba1.append((b, taxa))
            dpg.set_value("barra", (i+1)/len(blocos))
            dpg.split_frame()
            
        print(f"Tempo total: {time.time()-start_t:.2f}s\n") # LOG FINAL
            
        global plot_series_tags
        tag = f"plot1_{len(plot_series_tags)}"
        plot_series_tags.append(tag)
        x, y = zip(*res_aba1)
        x_log = [math.log2(v) for v in x]
        dpg.add_line_series(x_log, list(y), label=f"{algo}", parent="y_axis", tag=tag)
        dpg.fit_axis_data("x_axis")
        dpg.fit_axis_data("y_axis")
        dpg.set_value("resultados_box", f"Resultados Finais: {list(zip(x, [round(v,3) for v in y]))}")
    except Exception as e:
        print(f"Erro simulação: {e}")

def rodar_simulacao_multinivel_callback(sender, app_data):
    try:
        mem_size = dpg.get_value("memory_size_multi")
        acessos = dpg.get_value("acessos_multi")
        n_sim = dpg.get_value("n_simulacoes_multi")
        bloco = dpg.get_value("bloco_multi")
        
        niveis = []
        algos = []
        tempos = []
        
        niveis.append((dpg.get_value("tamanho_cache_l1")//bloco, dpg.get_value("associatividade_l1"), bloco))
        algos.append(dpg.get_value("algoritmo_l1"))
        tempos.append(dpg.get_value("tempo_l1"))
        
        if dpg.get_value("usar_l2"):
            niveis.append((dpg.get_value("tamanho_cache_l2")//bloco, dpg.get_value("associatividade_l2"), bloco))
            algos.append(dpg.get_value("algoritmo_l2"))
            tempos.append(dpg.get_value("tempo_l2"))
            
        if dpg.get_value("usar_l3"):
            niveis.append((dpg.get_value("tamanho_cache_l3")//bloco, dpg.get_value("associatividade_l3"), bloco))
            algos.append(dpg.get_value("algoritmo_l3"))
            tempos.append(dpg.get_value("tempo_l3"))
        
        tempos.append(dpg.get_value("tempo_ram"))
        probs = (dpg.get_value("prob_temporal_multi"), dpg.get_value("prob_espacial_multi"), dpg.get_value("prob_quente_multi"))
        regioes = [64, 1024, 8192, 32768, 131072]
        
        hit_rates, t_medio = simulacao_monte_carlo_multinivel_wrapper(n_sim, acessos, mem_size, niveis, algos, tempos, regioes, probs, bloco)
        
        msg = f"--- Resultados Multinível ---\nHit Rates (L1, L2...): {hit_rates}\nTempo Médio de Acesso: {t_medio:.2f} ns"
        dpg.set_value("resultados_box_multi", msg)
        
    except Exception as e:
        dpg.set_value("resultados_box_multi", f"Erro: {str(e)}")

# ------------------------------------------------------------------------------
# 6. CONSTRUÇÃO DA GUI (MAIN)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    dpg.create_context()
    
    with dpg.window(label="Simulador de Cache Educacional", tag="Primary Window"):
        with dpg.tab_bar(tag="tab_bar"):
            
            # --- ABA 1: CACHE ÚNICA ---
            with dpg.tab(label="Cache Única", tag="cache_unica_tab"):
                
                # PARTE SUPERIOR: INPUTS
                dpg.add_input_int(label="Memory Size", default_value=1048576, tag="memory_size", width=200)
                dpg.add_input_int(label="Acessos", default_value=10000, tag="acessos", width=200)
                dpg.add_input_int(label="Tamanho Cache (Bytes)", default_value=4096, tag="tamanho_cache", width=200)
                dpg.add_input_int(label="Associatividade", default_value=4, tag="associatividade", width=200)
                dpg.add_input_int(label="N Simulações", default_value=5, tag="n_simulacoes", width=200)
                dpg.add_separator()
                dpg.add_input_float(label="Prob. Temporal", default_value=0.3, tag="prob_temporal", width=200)
                dpg.add_input_float(label="Prob. Espacial", default_value=0.3, tag="prob_espacial", width=200)
                dpg.add_input_float(label="Prob. Quente", default_value=0.3, tag="prob_quente", width=200)
                dpg.add_separator()
                dpg.add_input_text(label="Tamanhos de Bloco", default_value="2,4,8,16,32,64,128,256,512", tag="blocos", width=400)
                dpg.add_separator()

                # BARRA DE BOTÕES
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Simular", callback=rodar_simulacao_aba1)
                    dpg.add_button(label="Simular Multinível (Ir p/ Aba 2)", callback=lambda: dpg.set_value(dpg.get_item_alias("tab_bar"), "cache_multinivel_tab")) 
                    dpg.add_button(label="Limpar Último", callback=limpar_ultimo_plot)
                    dpg.add_button(label="Limpar Plots", callback=limpar_plots)
                    dpg.add_button(label="Gerar Heatmap", callback=mapa_temporal_blocos_global) 
                    dpg.add_button(label="Limpar Heatmap", callback=limpar_heatmap)
                    dpg.add_button(label="Sair", callback=close_callback)
                    dpg.add_progress_bar(tag="barra", default_value=0.0, width=250)
                    dpg.add_text("0%", tag="texto")
                    dpg.add_combo(items=["FIFO", "LRU", "LFU", "Random"], default_value='FIFO', label="<-- Algoritmo", width=100, tag="combo_algoritmo",callback=selecionar_algoritmo)
                
                dpg.add_separator()
                dpg.add_input_text(label="<-- Resultado da Última Simulação", multiline=True, readonly=True, height=35, tag="resultados_box")
                dpg.add_text("", tag="mensagem_erro", color=(255, 100, 100))
                dpg.add_separator()

                # ÁREA INFERIOR: GRÁFICOS
                with dpg.group(horizontal=True):
                    with dpg.plot(label="Taxa de acerto vs Tamanho do Bloco", tag="plot", height=380, width=600):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Tamanho do Bloco", tag="x_axis")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Taxa de Acerto", tag="y_axis")
                    
                    # Coluna da direita: Log + Heatmap
                    with dpg.group():
                        dpg.add_input_text(label="<-- Resumo", multiline=True, readonly=True, height=380, width=360, tag="Resumo")
                        with dpg.group(tag="heatmap_group"):
                            dpg.add_text("Mapa de Calor (Gerado aqui)")

            # --- ABA 2: MULTINÍVEL ---
            with dpg.tab(label="Multinível", tag="cache_multinivel_tab"):
                with dpg.group(horizontal=True):
                    with dpg.group(width=350):
                        dpg.add_text("Configurações Multinível", color=(255, 200, 100))
                        dpg.add_input_int(label="Memory Size", default_value=16777216, tag="memory_size_multi", width=150)
                        dpg.add_input_int(label="Acessos", default_value=50000, tag="acessos_multi", width=150)
                        dpg.add_input_int(label="N Simulações", default_value=5, tag="n_simulacoes_multi", width=150)
                        dpg.add_separator()
                        
                        dpg.add_text("L1 Cache")
                        dpg.add_input_int(label="Tam L1", default_value=32768, tag="tamanho_cache_l1", width=150)
                        dpg.add_input_int(label="Assoc L1", default_value=8, tag="associatividade_l1", width=150)
                        dpg.add_combo(items=["FIFO", "LRU", "LFU", "Random"], default_value='LRU', tag="algoritmo_l1", width=150)
                        dpg.add_input_int(label="Tempo L1", default_value=1, tag="tempo_l1", width=150)
                        dpg.add_separator()
                        
                        dpg.add_checkbox(label="Usar L2", default_value=True, tag="usar_l2")
                        dpg.add_input_int(label="Tam L2", default_value=262144, tag="tamanho_cache_l2", width=150)
                        dpg.add_input_int(label="Assoc L2", default_value=8, tag="associatividade_l2", width=150)
                        dpg.add_combo(items=["FIFO", "LRU", "LFU", "Random"], default_value='LRU', tag="algoritmo_l2", width=150)
                        dpg.add_input_int(label="Tempo L2", default_value=10, tag="tempo_l2", width=150)
                        dpg.add_separator()
                        
                        dpg.add_checkbox(label="Usar L3", default_value=True, tag="usar_l3")
                        dpg.add_input_int(label="Tam L3", default_value=2097152, tag="tamanho_cache_l3", width=150)
                        dpg.add_input_int(label="Assoc L3", default_value=16, tag="associatividade_l3", width=150)
                        dpg.add_combo(items=["FIFO", "LRU", "LFU", "Random"], default_value='LRU', tag="algoritmo_l3", width=150)
                        dpg.add_input_int(label="Tempo L3", default_value=30, tag="tempo_l3", width=150)
                        dpg.add_separator()
                        
                        dpg.add_input_int(label="Tempo RAM", default_value=200, tag="tempo_ram", width=150)
                        dpg.add_input_float(label="Prob Temporal", default_value=0.3, tag="prob_temporal_multi", width=150)
                        dpg.add_input_float(label="Prob Espacial", default_value=0.3, tag="prob_espacial_multi", width=150)
                        dpg.add_input_float(label="Prob Quente", default_value=0.3, tag="prob_quente_multi", width=150)
                        dpg.add_input_int(label="Tam Bloco", default_value=64, tag="bloco_multi", width=150)
                        
                        dpg.add_spacer(height=20)
                        dpg.add_button(label="RODAR SIMULAÇÃO MULTINÍVEL", callback=rodar_simulacao_multinivel_callback, width=300, height=50)

                    with dpg.group():
                        dpg.add_text("Resultados da Simulação Multinível:")
                        dpg.add_input_text(tag="resultados_box_multi", multiline=True, width=600, height=400, readonly=True)

            # --- ABA 3: ANIMAÇÃO ---
            with dpg.tab(label="Simulação Métodos de Escrita"):
                dpg.add_text("Comparação Visual: Write-Through vs Write-Back", color=(100, 255, 100))
                dpg.add_text("Observe o barramento (linha) conectando Cache à RAM.")
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="> INICIAR", callback=start_anim_callback, width=150)
                    dpg.add_button(label="|| PAUSAR", callback=pause_anim_callback, width=100)
                    dpg.add_button(label="[] REINICIAR", callback=reset_anim_callback, width=100)
                    dpg.add_text(" Velocidade:")
                    dpg.add_slider_float(default_value=10, min_value=1, max_value=100, width=150, callback=update_speed_callback)
                
                dpg.add_progress_bar(tag="anim_progress", default_value=0.0, width=600)
                dpg.add_spacer(height=20)
                
                with dpg.drawlist(width=800, height=400, tag="canvas_anim"):
                    # WT
                    dpg.draw_text((20, 20), "CENÁRIO A: WRITE-THROUGH (Imediata)", color=(255, 255, 0), size=20)
                    dpg.draw_rectangle((50, 60), (150, 140), color=(100, 200, 255), thickness=3) 
                    dpg.draw_text((70, 90), "CPU", size=20)
                    dpg.draw_rectangle((250, 60), (350, 140), color=(100, 255, 100), thickness=3) 
                    dpg.draw_text((260, 90), "CACHE", size=18)
                    dpg.draw_rectangle((550, 60), (650, 140), color=(255, 100, 255), thickness=3) 
                    dpg.draw_text((570, 90), "RAM", size=20)
                    dpg.draw_line((150, 100), (250, 100), color=(255, 255, 255), thickness=2)
                    dpg.draw_line((350, 100), (550, 100), color=(50, 50, 50), thickness=2, tag="wt_bus_line")
                    dpg.draw_text((400, 70), "Barramento", size=15)
                    
                    # WB
                    dpg.draw_text((20, 220), "CENÁRIO B: WRITE-BACK (Posterior)", color=(255, 100, 100), size=20)
                    dpg.draw_rectangle((50, 260), (150, 340), color=(100, 200, 255), thickness=3)
                    dpg.draw_text((70, 290), "CPU", size=20)
                    dpg.draw_rectangle((250, 260), (350, 340), color=(100, 255, 100), thickness=3)
                    dpg.draw_text((260, 290), "CACHE", size=18)
                    dpg.draw_rectangle((550, 260), (650, 340), color=(255, 100, 255), thickness=3)
                    dpg.draw_text((570, 290), "RAM", size=20)
                    dpg.draw_line((150, 300), (250, 300), color=(255, 255, 255), thickness=2)
                    dpg.draw_line((350, 300), (550, 300), color=(50, 50, 50), thickness=2, tag="wb_bus_line")

                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text("Status WT:", color=(255, 255, 0))
                        dpg.add_text("Pronto", tag="wt_status", wrap=350)
                        dpg.add_spacer(height=10)
                        dpg.add_text("ESCRITAS NA RAM (Custo):")
                        dpg.add_text("0", tag="wt_counter", color=(255, 255, 0))
                    dpg.add_spacer(width=50)
                    with dpg.group():
                        dpg.add_text("Status WB:", color=(255, 100, 100))
                        dpg.add_text("Pronto", tag="wb_status", wrap=350)
                        dpg.add_spacer(height=10)
                        dpg.add_text("ESCRITAS NA RAM (Custo):")
                        dpg.add_text("0", tag="wb_counter", color=(255, 100, 100))

    dpg.create_viewport(title='Simulador de Cache', width=1300, height=900)
    dpg.set_primary_window("Primary Window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    # Redireciona o print para a caixa de texto da Aba 1
    sys.stdout = DPGRedirector("Resumo")
    
    while dpg.is_dearpygui_running():
        update_animation() # Chama a animação a cada frame
        dpg.render_dearpygui_frame()
    
    dpg.destroy_context()