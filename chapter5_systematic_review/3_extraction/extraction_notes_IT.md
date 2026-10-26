<!-- Extraction notes (original language: Italian). One entry per study, with the PDF page of every
verified statement. 'Tracker' refers to the author's preliminary screening log; entries marked
CORREZIONE record corrections of bibliographic or factual details found on the full text.
Study identifiers here are the legacy identifiers (column legacy_id of coding_sheet.csv). -->

# Note di estrazione (verificate sui PDF)

Formato di ogni scheda: riferimento bibliografico · famiglia e architettura · dati · task · baseline · metriche · fatti stilizzati (SF) testati · codice · rubrica Q1–Q7 · affermazioni verificate, con pagina del PDF · eventuali correzioni rispetto al tracker.

**Rubrica di qualità**
- Q1 dati pubblici
- Q2 codice disponibile
- Q3 baseline classica (GARCH, bootstrap, SDE o modello parametrico)
- Q4 baseline deep
- Q5 split out-of-sample temporale per il task a valle
- Q6 diagnostica dei fatti stilizzati
- Q7 più seed, deviazioni standard o test di significatività

Valori: Y / N / P (parziale) / NA (non applicabile).

**Codici dei fatti stilizzati**
- SF1 code pesanti
- SF2 assenza di autocorrelazione lineare
- SF3 volatility clustering
- SF4 long-range dependence
- SF5 leverage effect
- SF6 asimmetria gain/loss e skewness
- SF7 aggregational Gaussianity
- SF8 correlazione coarse-fine della volatilità
- SF9 dipendenza cross-asset
- SF10 stagionalità intraday e fatti di microstruttura
- SF11 mean reversion e regimi

**Categorie di metriche**
- MB model-based (discriminative/predictive score, TSTR)
- FB feature-based (momenti, ACF)
- DB distance-based (W1, MMD, KS, DTW, FID)
- VIS visualizzazione
- PATH metriche sullo spazio dei cammini (Sig-MMD, Sig-W1)
- CAUS metriche causali o adattate alla filtrazione
- RISK metriche di rischio (VaR, ES)
- DEC metriche decisionali e di trading
- PROB metriche probabilistiche (CRPS, NLL)
- EFF efficienza computazionale

---

## Cartella 01 — Dipendenza temporale

### Yoon, Jarrett & van der Schaar (2019) — TimeGAN
- **Bib:** NeurIPS 2019 (peer-reviewed).
- **Modello:** GAN ibrida, con reti di embedding e recovery (autoencoder) più generatore e discriminatore nello spazio latente, tutte RNN. Combina una loss supervisionata step-wise con la loss avversaria.
- **Dati:** Sines, Stocks (Google giornaliero 2004–2019), Energy, Events. **Task:** dati sintetici.
- **Baseline:** RCGAN, C-RNN-GAN, T-Forcing, P-Forcing, WaveNet, WaveGAN. **Metriche:** VIS (t-SNE, PCA); MB (discriminative score, predictive score TSTR). **SF:** nessuno.
- **Codice:** Y (github.com/jsyoon0823/TimeGAN, citato da TsT-GAN a p.7).
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 Y, Q5 NA, Q6 N, Q7 Y.
- **Affermazioni verificate:**
  - introduce il discriminative score (classificatore post-hoc, |0.5 − accuratezza|) e il predictive score (TSTR) (p.6);
  - miglioramento statisticamente significativo rispetto a RCGAN (p.8);
  - l'ablazione mostra che la loss supervisionata conta soprattutto sui dati ad alta correlazione temporale come Stocks (p.9).

### Wiese, Knobloch, Korn & Kretschmer (2020) — Quant GANs
- **Bib:** Quantitative Finance 20(9), 2020. Nel fascicolo c'è la versione arXiv 1907.06673v2; **va citata la versione su QF** (NOTE.txt).
- **Modello:** GAN con generatore e discriminatore basati su TCN. Una trasformazione Lambert W inversa gestisce le code pesanti. Il "log-return neural process" (NP) è costruito in modo da ammettere il passaggio alla misura risk-neutral.
- **Dati:** S&P 500 giornaliero, maggio 2009 – dicembre 2018. **Task:** dati sintetici.
- **Baseline:** GARCH(1,1) con drift costante; varianti "pure TCN" e "constrained log-return NP".
- **Metriche:** DB (W1 sulle distribuzioni dei rendimenti a 1, 5, 20 e 100 giorni); FB (ACF score su rendimenti, rendimenti assoluti e al quadrato; leverage score) (p.18–19). **SF:** SF1, SF2, SF3, SF5, SF7 (W1 su più orizzonti).
- **Codice:** N (nessun repository indicato nel testo).
- **Q:** Q1 Y, Q2 N, Q3 Y, Q4 P, Q5 NA, Q6 Y, Q7 N.
- **Affermazioni verificate:**
  - il modello riproduce proprietà distributive e di dipendenza "far better" del GARCH (p.17);
  - la Lambert W è essenziale per le code pesanti (p.3, p.16);
  - le conclusioni individuano due sfide aperte (p.20): (i) modellare ed estrapolare la coda con conoscenza a priori (indice di coda); (ii) **una metrica unica** che unifichi le metriche distributive e quelle di dipendenza. Sono utili per la Sez. 6.
- **Nota:** l'abstract dichiara di catturare "long-range dependencies such as volatility clusters", in contrasto con He & Rachev 2025.

### Pei, Ren, Yang, Liu, Qin & Li (2021) — RTSGAN
- **Bib:** IEEE ICDM 2021, DOI 10.1109/ICDM51629.2021.00058.
- **Modello:** un autoencoder encoder-decoder comprime l'intera serie in un latente di dimensione fissa, e una WGAN genera in quello spazio latente. La variante RTSGAN-M gestisce i dati mancanti.
- **Dati:** Stocks (Google), Energy, PhysioNet, MIMIC-III. **Task:** dati sintetici con lunghezza variabile e dati mancanti.
- **Baseline:** TimeGAN, RCGAN, C-RNN-GAN, COT-GAN, DoppelGANger e altre. **Metriche:** MB (discriminative e predictive score, TSTR); VIS. **SF:** nessuno.
- **Codice:** Y (seqml.github.io/rtsgan, p.1).
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 Y, Q5 NA, Q6 N, Q7 P.

### Fu, Hirsa & Osterrieder (2022) — GAN con attention (TAGAN, TTGAN)
- **Bib:** arXiv 2207.00493 (preprint).
- **Modello:** TAGAN (convoluzioni più attention causale) e TTGAN (Transformer con attention sparsa).
- **Dati:** S&P 500 giornaliero (lo stesso campione di QuantGAN, maggio 2009 – novembre 2018); un dataset a curtosi media; superficie di volatilità delle opzioni su S&P 500. **Task:** dati sintetici.
- **Baseline:** QuantGAN.
- **Metriche:** DB (W1 sui rendimenti a 1–200 giorni); FB (skewness, curtosi, ACF, ACF dei valori assoluti, ACF dei quadrati, leverage, cross-correlazione); per le opzioni, il **tasso di arbitraggio**. **SF:** SF1, SF2, SF3, SF4 (ACF di |r| su lag lunghi), SF5, SF6, SF9.
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 Y, Q5 NA, Q6 Y, Q7 N.
- **Affermazioni verificate:**
  - TTGAN genera un'ACF dei rendimenti più regolare, mentre QuantGAN fa overfitting delle curve di ACF (p.18);
  - sulle superfici di opzioni QuantGAN non riproduce l'autocorrelazione dei rendimenti di volatilità (p.18).
- **CORREZIONE rispetto al tracker:** sulle superfici IV i **tassi di arbitraggio restano elevati**: QuantGAN 30,10%, TAGAN tra 1,55% e 21,16%, TTGAN tra 8,88% e 12,86% (Tab. 5, p.18). Il tracker dice invece "mantenendo le relazioni di arbitraggio statico". È un dato utile per la Sez. 5.4.3: la regolarità ottenuta con la sola attention non garantisce l'assenza di arbitraggio.

### Srinivasan & Knottenbelt (2022) — TsT-GAN
- **Bib:** arXiv 2205.11164 (preprint).
- **Modello:** GAN basata su Transformer (embedder, predictor, generatore, discriminatore), con il generatore addestrato tramite masked modelling e una loss ausiliaria di moment matching.
- **Dati:** Sines, Stocks, Energy, Chickenpox, Air. **Task:** dati sintetici.
- **Baseline:** TimeGAN, RCGAN, C-RNN-GAN, COT-GAN, P-Forcing. **Metriche:** MB (predictive e discriminative score, con deviazioni standard); VIS. **SF:** nessuno.
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 Y, Q5 NA, Q6 N, Q7 Y.
- **Affermazioni verificate:**
  - miglior predictive score su tutti i dataset (p.7);
  - **su Stocks un generatore autoregressivo batte TsT-GAN nel discriminative score** (p.7);
  - limite dichiarato: complessità O(N²) della self-attention (p.8).

### Liu et al. (2023/2024) — Time-Transformer AAE
- **Bib:** il PDF caricato è la submission anonima a ICLR 2023 (double-blind). **Da citare nella versione pubblicata**, se esiste (credo SDM 2024 con titolo "Time-Transformer: Integrating Local and Global Features for Better Time Series Generation"; da verificare).
- **Modello:** autoencoder avversario (AAE) con decoder "Time-Transformer": rami TCN e Transformer in parallelo, fusi da una cross-attention bidirezionale.
- **Dati:** Stock (Google giornaliero, 3.686 sequenze), sinusoidi simulate, dati medici e altri. **Task:** dati sintetici, più augmentation per la classificazione sbilanciata.
- **Baseline:** TimeVAE, RTSGAN, TimeGAN, RCGAN e altre.
- **Metriche:** DB (FID calcolata con TS2Vec, cioè una Context-FID); MB (discriminative e predictive score, con IC al 95%); VIS. **SF:** nessuno.
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 Y, Q5 NA, Q6 N, Q7 Y.
- **Affermazione verificata:** migliore in tutti i dataset **tranne "Stock"** per il discriminative score (p.8).

### Zhou, Poli, Xu, Massaroli & Ermon (2023) — LS4
- **Bib:** ICML 2023, PMLR 202 (p.1). Non è solo arXiv come indicato nel tracker.
- **Modello:** VAE sequenziale con dinamica latente data da una state-space ODE (blocchi di tipo S4) e rappresentazione convoluzionale.
- **Dati:** FRED-MD (macroeconomico-finanziario), NN5 Daily, Temperature Rain, Solar Weekly; per i dati irregolari Physionet e USHCN. **Task:** dati sintetici, interpolazione ed estrapolazione.
- **Baseline:** Latent ODE, ODE-RNN, RNN-VAE, SaShiMi, TimeGAN e altre.
- **Metriche:** DB (distanza tra marginali); MB (classification e prediction score); PROB (MSE e CRPS per interpolazione ed estrapolazione); EFF (tempi di esecuzione). **SF:** nessuno.
- **Codice:** da verificare.
- **Q:** Q1 Y, Q2 ?, Q3 N, Q4 Y, Q5 NA, Q6 N, Q7 P.
- **Affermazioni verificate:**
  - circa 100 volte più veloce delle baseline sulle sequenze lunghe (p.2 e p.8);
  - le baseline faticano con le transizioni brusche ("stiff") di NN5 Daily e Temperature Rain (p.7).

### Naiman, Erichson, Ren, Mahoney & Azencot (2024a) — KoVAE
- **Bib:** ICLR 2024.
- **Modello:** VAE con prior latente lineare ispirato all'operatore di Koopman; è possibile imporre vincoli spettrali sugli autovalori.
- **Dati:** Sines, Stocks (Google 2004–2019), Energy, MuJoCo, in configurazione regolare e irregolare. **Task:** dati sintetici, anche con campionamento irregolare.
- **Baseline:** TimeGAN, GT-GAN e altre. **Metriche:** MB (discriminative e predictive score); VIS (t-SNE, densità). **SF:** nessuno.
- **Codice:** Y (GitHub, p.6).
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 Y, Q5 NA, Q6 N, Q7 P.

### Desai, Freeman, Wang & Beaver (2021) — TimeVAE
- **Bib:** arXiv 2111.08095 (submission a ICLR 2022 non accettata, quindi preprint).
- **Modello:** VAE con encoder e decoder convoluzionali; la variante interpretabile ha un decoder con blocchi separati di trend e stagionalità.
- **Dati:** stockv (titolo azionario Yahoo, 3.919 campioni giornalieri a 6 dimensioni), sines, energy, air. **Task:** dati sintetici.
- **Baseline:** TimeGAN, RCGAN, T-Forcing. **Metriche:** VIS (t-SNE); MB (discriminative e predictive score, con IC). **SF:** nessuno.
- **Codice:** Y (github.com/abudesai/syntheticdatagen, p.6).
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 Y, Q5 NA, Q6 N, Q7 Y.

### Chen, Feng, Zhang, Xiao, Gao & Zhao (2024) — SDformer
- **Bib:** NeurIPS 2024 (p.1).
- **Modello:** tokenizer con vector quantization guidata dalla similarità, seguito da un Transformer discreto con modellazione dei token autoregressiva e mascherata.
- **Dati:** Sines, Stocks, ETTh, MuJoCo, Energy e altri. **Task:** dati sintetici, incondizionati e condizionati.
- **Baseline:** Diffusion-TS, TimeGAN, TimeVAE e altre.
- **Metriche:** MB (discriminative e predictive score); DB (Context-FID); VIS (t-SNE, KDE); EFF (tempo di inferenza). **SF:** nessuno.
- **Codice:** da verificare.
- **Q:** Q1 Y, Q2 ?, Q3 N, Q4 Y, Q5 NA, Q6 N, Q7 Y.
- **Affermazioni verificate:**
  - miglioramento medio del 60,8% nel discriminative score e dell'86,5% nella Context-FID (p.2);
  - motivazione dichiarata: tempi di inferenza "excessively long" dei DDPM (p.1).

### Alaa, Chan & van der Schaar (2021) — Fourier Flows
- **Bib:** ICLR 2021.
- **Modello:** normalizing flow nel dominio delle frequenze (DFT con filtri spettrali); verosimiglianza esatta.
- **Dati:** sinusoidi sintetiche, Google Stocks, UCI Energy, dati clinici (Lung). **Task:** dati sintetici.
- **Baseline:** TimeGAN, WGAN, RCGAN, WaveNet, RealNVP.
- **Metriche:** MB (predictive score); DB (precision e recall); PSD (densità spettrale di potenza); verosimiglianza; VIS. **SF:** nessuno.
- **Codice:** da verificare.
- **Q:** Q1 Y, Q2 ?, Q3 N, Q4 Y, Q5 NA, Q6 N, Q7 P.
- **Affermazione verificata:** Fourier Flows supera tutti i metodi nel predictive score su Stocks e Lung (p.8).

### Yuan & Qiao (2024) — Diffusion-TS
- **Bib:** ICLR 2024.
- **Modello:** DDPM con Transformer encoder-decoder e decomposizione trend/stagionalità; ricostruisce direttamente il segnale invece di stimare il rumore, con una loss nel dominio di Fourier.
- **Dati:** Sines, Stocks (Google 2004–2019), ETTh, MuJoCo, Energy, fMRI. **Task:** dati sintetici, imputazione, forecasting.
- **Baseline:** TimeGAN, TimeVAE, Diffwave, DiffTime, Cot-GAN e altre.
- **Metriche:** MB (discriminative e predictive score); DB (Context-FID, correlational score); VIS. **SF:** nessuno.
- **Codice:** Y (github.com/Y-debug-sys/Diffusion-TS, p.1).
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 Y, Q5 NA, Q6 N, Q7 Y (5 run).
- **Affermazione verificata:** discriminative score migliorato in media del 50% sui sei dataset (p.7).

### Lim, Kim, Park & Park (2023) — TSGM (SGM per serie regolari)
- **Bib:** arXiv 2301.08518 ("under review"; preprint).
- **Modello:** score-based generative model (SDE VP o subVP) nello spazio latente di un autoencoder RNN, con conditional score network.
- **Dati:** Stocks, Energy, AI4I, Occupancy e altri. **Task:** dati sintetici.
- **Baseline:** 8, tra cui TimeGAN, TimeVAE e GT-GAN. **Metriche:** MB (discriminative e predictive score); VIS. **SF:** nessuno.
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 Y, Q5 NA, Q6 N, Q7 Y (10 seed).
- **Affermazione verificata:** limite dichiarato: gli SGM sono "slower than GANs" nel campionamento (p.7).

### Naiman, Berman, Pemper, Arbiv, Fadlon & Azencot (2024b) — ImagenTime
- **Bib:** NeurIPS 2024 (da confermare sul PDF; il codice è indicato).
- **Modello:** trasformazioni invertibili delle serie in immagini (delay embedding, STFT) e backbone di diffusion per la visione (EDM).
- **Dati:** sequenze brevi (Stocks e altri); lunghe (FRED-MD, NN5 Daily, Temperature Rain); ultra-lunghe (USHCN e altri). **Task:** dati sintetici, interpolazione ed estrapolazione.
- **Baseline:** DiffTime, LS4, TimeGAN, GT-GAN e altre.
- **Metriche:** MB (discriminative e predictive score); DB (marginali; distanza di Wasserstein tra embedding t-SNE); VIS. **SF:** nessuno.
- **Codice:** Y (github.com/azencot-group/ImagenTime, p.1).
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 Y, Q5 NA, Q6 N, Q7 P.
- **Affermazione verificata:** limite dichiarato: richiede "slightly higher computational resources" (p.9).

### He & Rachev (2025/2026) — LRD e Quant GANs ★ (studio critico)
- **Bib:** il PDF caricato è arXiv 2509.19663v3 (25 giugno 2026). **Da citare la versione su Studies in Nonlinear Dynamics & Econometrics** (NOTE.txt).
- **Modello valutato:** Quant GAN (TCN).
- **Dati:** 9 asset giornalieri, 3.319 osservazioni: S&P 500, DAX, Nikkei 225; frumento, mais, soia; UNG, USO, XLE (p.1, p.4). **Task:** valutazione critica della capacità di riprodurre la long-range dependence.
- **Baseline:** ARMA(1,1)-GARCH(1,1), Gaussiana iid, bootstrap storico iid, moving-block bootstrap con blocchi di 20 (p.21, p.23).
- **Metriche:** DB (W1); RISK (errore sui quantili di coda all'1%, 5%, 95% e 99%); FB (errore sull'ACF di |r| ai lag 1–20); Hurst stimato con R/S e DFA; ARFIMA-FIGARCH. **SF:** SF1, SF3, SF4.
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 Y, Q4 N, Q5 NA, Q6 Y, Q7 Y (ensemble di 10.000 traiettorie).
- **Affermazioni verificate:**
  - la long memory è marcata nella volatilità condizionata e limitata nei rendimenti medi (p.1, p.29);
  - **Tab. 6 (p.23):** Quant GAN batte la Gaussiana iid in W1 ma **non domina i benchmark classici**:
    - errore sui quantili di coda 4,74·10⁻³, contro 0,96·10⁻³ del bootstrap storico;
    - errore sull'ACF della volatilità 0,152, contro 0,060 di ARMA-GARCH;
    - errori sull'esponente di Hurst (R/S e DFA) i più alti;
  - **Tab. 7 (p.24):** su 10.000 traiettorie generate, la mediana dell'esponente di Hurst è **sotto il valore empirico per tutti e 9 gli asset**, spesso con anche il quantile al 95% sotto il valore reale. Per l'S&P 500 con R/S: 0,506 reale contro 0,442 generato. L'attenuazione della memoria è quindi sistematica;
  - implicazione dichiarata dagli autori: rischio sottostimato su orizzonti lunghi e benefici di diversificazione troppo ottimistici (p.29–30).
- **CORREZIONE rispetto al tracker:** gli asset non sono S&P 500, Dow Jones e Nasdaq ma i 9 indicati sopra (versione v3). Inoltre il risultato più forte è il confronto con i benchmark classici, che il tracker non menziona.

### Kwon & Lee (2024) — Can GANs Learn the Stylized Facts? ★ (studio critico)
- **Bib:** ACM ICAIF '24 (p.1).
- **Modello:** GAN vanilla con generatori di diversa architettura (MLP, MLP-CNN, LSTM, GRU, TCN), addestrata con una loss JSD sulla distribuzione dei rendimenti e del valore finale.
- **Dati:** dati simulati da 5 processi (BM, GBM, OU, jump-diffusion, Heston) più una GBM bivariata con correlazione 0,9 e 0. **Task:** valutazione critica.
- **Baseline:** la "ground truth" simulata. **Metriche:** FB (statistiche dei parametri: velocità di reversione, salti, vol-of-vol, correlazioni). **SF:** SF1, SF9, SF11 (mean reversion).
- **Codice:** N.
- **Q:** Q1 Y (simulato), Q2 N, Q3 Y (come ground truth), Q4 P, Q5 NA, Q6 Y, Q7 P.
- **Affermazioni verificate (p.8):**
  - nel caso univariato le GAN catturano le strutture temporali generali ma "struggle with detailed features";
  - BM e GBM sono appresi bene, mentre **il generatore TCN fatica a riprodurre la velocità di reversione dell'OU**;
  - solo MLP e MLP-CNN catturano la distribuzione bimodale dei salti;
  - per Heston la vol-of-vol generata è più platicurtica, e il problema peggiora al crescere della vol-of-vol;
  - nel caso multivariato l'errore sulle correlazioni è contenuto (0,074 e 0,057), ma il modello "struggles to accurately reproduce the individual distributions and dependencies".

---

## Cartella 02 — Code pesanti ed eventi estremi

### Han, Ma & Zhang (2022) — WGAN-GP multi-asset
- **Bib:** Journal of Risk and Financial Management 15, 26 (2022).
- **Modello:** WGAN-GP.
- **Dati:** titoli azionari del Hang Seng Index Composite; futures su metalli preziosi (CME, TOCOM/JPX); criptovalute spot e perpetual (BTC, ETH, BNB su Binance). **Tutti a frequenza intraday, da 1 a 15 minuti.** **Task:** dati sintetici.
- **Baseline:** confronto qualitativo con VAE (KS 0,218 contro 0,126 su GOOG, p.10, da letteratura).
- **Metriche:** DB (MAE, MSE e RMSE tra serie reali e generate; statistica KS; rapporto KS R/F; MAE-R per tick). FB descrittivo (media, deviazione standard, skewness, curtosi). **SF:** SF1 solo in forma descrittiva.
- **Codice:** N.
- **Q:** Q1 P, Q2 N, Q3 N, Q4 N, Q5 NA, Q6 P, Q7 N.
- **CORREZIONE rispetto al tracker:** il tracker cita S&P 500, DJIA, WTI, oro e Bitcoin a frequenza giornaliera, con generatore LSTM e discriminatore CNN. Il PDF riporta invece dati HK, futures su metalli e cripto **a frequenza minuto**, e nessuna analisi di ACF o volatility clustering. Gli autori indicano l'S&P 500 come estensione futura (p.18–19).

### Kang (2024/2025) — WGAN-BiLSTM per code pesanti in regime few-shot
- **Bib:** Computational Economics 65:1775–1794 (2025), DOI 10.1007/s10614-024-10639-9.
- **Modello:** una WGAN con layer fully connected per l'augmentation, seguita da un BiLSTM per la previsione.
- **Dati:** Ping An Bank (1.331 giorni) e Haier Smart Home (1.184 campioni), con augmentation del 30%, 60% e 100%. **Task:** augmentation per il forecasting.
- **Baseline:** GAN, DCGAN; LSTM, GRU e BiLSTM senza augmentation.
- **Metriche:** DB (statistica KS); area tra i grafici log-log delle code; PROB/err (MAE, RMSE, R²). **SF:** SF1.
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 Y, Q5 Y (split temporale), Q6 P, Q7 N.
- **Affermazione verificata:** l'area tra le code è nettamente minore con WGAN (circa 3–4 su (a)) che con GAN (circa 15–16) o DCGAN (circa 8–10) (Tab. 8, p.11).

### Noguer i Alonso & Pacheco Aznar (2023) — VAE univariato
- **Bib:** working paper SSRN, AIFI (preprint).
- **Modello:** VAE.
- **Dati:** rendimenti dell'S&P 500 e traiettorie simulate da Heston. **Task:** dati sintetici univariati.
- **Baseline:** nessuna. **Metriche:** DB (MMD); confronto descrittivo di momenti e distribuzioni. **SF:** SF1 in forma descrittiva.
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 P (Heston come ground truth), Q4 N, Q5 NA, Q6 P, Q7 N.
- **Affermazione verificata:** MMD bassa, ma le distribuzioni presentano differenze visibili. Gli autori avvertono sulla non-stazionarietà e rimandano il caso multivariato a lavori futuri (p.1, p.5, p.8).
- **CORREZIONE rispetto al tracker:** il tracker parla di test KS. Nel PDF il test usato è la **MMD**.

### Kim, Choi & Kim (2025) — diffusion con forward process GBM
- **Bib:** arXiv 2507.19003 (luglio 2025, preprint).
- **Modello:** score-based diffusion con rumore moltiplicativo di tipo GBM nel forward process. Nello spazio dei log-prezzi si riduce a una SDE VE. Il denoiser è un Transformer derivato da CSDI.
- **Dati:** costituenti dell'S&P 500, giornalieri. **Task:** dati sintetici.
- **Baseline:** diffusion standard VE e VP; il confronto con le GAN avviene tramite la letteratura. **Metriche:** FB (code in legge di potenza, ACF di |r|, funzione di leverage). **SF:** SF1, SF3, SF5.
- **Codice:** da verificare.
- **Q:** Q1 Y, Q2 ?, Q3 N, Q4 Y, Q5 NA, Q6 Y, Q7 N.
- **Affermazioni verificate:**
  - con capacità limitata il modello riproduce clustering e code ma **non il leverage effect**, che richiede un embedding delle feature più ampio (p.9 e p.11);
  - limite dichiarato: si discosta dalla log-normalità esatta delle marginali (p.16–17).

### Galib, Tan & Luo (2024) — FIDE
- **Bib:** NeurIPS 2024 (p.1).
- **Modello:** diffusion condizionale con inflazione delle alte frequenze nel dominio di Fourier, condizionamento sui block maxima e loss GEV.
- **Dati:** dati sintetici; Stocks (Google 2004–2019, prezzi di chiusura aggiustati); temperature; ECG e altri. **Task:** dati sintetici attenti agli estremi.
- **Baseline:** cGAN, TimeGAN, β-VAE, CVAE, TimeVAE, DDPM e altre diffusion.
- **Metriche:** DB (divergenze JS e KL sui block maxima); PROB (CRPS); MB (predictive score). **SF:** SF1 (estremi e block maxima).
- **Codice:** Y (github.com/galib19/FIDE, p.7).
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 Y, Q5 NA, Q6 P, Q7 Y.
- **Affermazioni verificate:**
  - la rimozione delle alte frequenze cancella gli estremi (p.3);
  - FIDE è migliore sulle metriche dei block maxima, mentre **TimeGAN e TimeVAE sono marginalmente migliori nel predictive score** (p.9–10);
  - c'è un trade-off tra estremi e utilità predittiva.

### Rosolia & Osterrieder (2021) [R] — GAN vanilla multi-asset
- **Bib:** preprint arXiv/ZHAW (1 agosto 2021).
- **Modello:** GAN vanilla con layer densi.
- **Dati:** Apple, S&P 500, VIX futures, oro, EUR/USD, giornalieri 2010–2021. **Task:** dati sintetici.
- **Baseline:** nessuna. **Metriche:** FB (skewness, curtosi su 1.000 distribuzioni sintetiche; ACF dei rendimenti e di |r|). **SF:** SF1, SF2, SF3.
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 N, Q5 NA, Q6 Y, Q7 P.
- **Affermazioni verificate:**
  - curtosi di Apple: 6,4 reale contro 1,18 sintetica (p.10–11);
  - l'ACF differisce nei primi 60 lag (p.14);
  - gli autori suggeriscono architetture ricorrenti come lavoro futuro (p.15).

### Pfenninger, Bigler, Rikli & Osterrieder (2021) [R] — WGAN-GP con LSTM su Bitcoin
- **Bib:** arXiv 2107.06008 (preprint).
- **Modello:** WGAN-GP con generatore e critic LSTM.
- **Dati:** Bitcoin giornaliero (2014–2021), confrontato con l'S&P 500. **Task:** dati sintetici.
- **Baseline:** nessuna. **Metriche:** FB (QQ-plot, skewness, curtosi, ACF). **SF:** SF1, SF3.
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 N, Q5 NA, Q6 Y, Q7 N.
- **Affermazione verificata:** le code estreme sono sottorappresentate, con distribuzioni meno volatili dell'originale (da riscontrare in Sez. 6.3–6.5 del PDF).

---

## Cartella 03 — Dipendenza cross-asset e alta dimensionalità

### Caulfield & Gleeson (2024) — confronto sistematico tra modelli deep (DGM) e modelli parametrici multivariati ★
- **Bib:** arXiv 2412.06417 (preprint).
- **Modelli confrontati:**
  - deep: RCGAN, TimeGAN, GMMN, CoMeTS, CTVAE (VAE con struttura AR-FNN), CTNF (normalizing flow);
  - parametrici: DCC-GARCH con innovazioni normali o t, Copula-GARCH, Factor Stochastic Volatility e le relative varianti a finestra mobile.
- **Dati:**
  - sintetici di complessità crescente: NGARCH e Heston, con l'aggiunta di regimi e salti (NGARCH+, Heston+);
  - empirici: 50 titoli dell'S&P 500, chiusure giornaliere, con opzioni.
- **Task:** dati sintetici e un'applicazione al trading di volatilità implicita (straddle long/short) tramite un modello HAR alimentato da feature generative.
- **Metriche:** DB (EMD sulle distribuzioni di momenti, autocorrelazioni e correlazioni, sia sull'intera serie sia su finestre mobili); reti di correlazione con indice di Jaccard; DEC (P&L degli straddle). **SF:** SF1, SF3, SF9, SF11 (regimi).
- **Codice:** N.
- **Q:** Q1 P, Q2 N, Q3 Y, Q4 Y, Q5 Y (split 60/10/30), Q6 Y, Q7 Y (5 seed).
- **Affermazioni verificate:**
  - **ranking combinato (Tab. 6, p.5–6):** RCGAN 4,80, FSVR 5,45, DCC-t 5,85, GMMN 5,95 … CTVAE ultimo con 12,60. Il miglior modello deep vince, ma **due modelli parametrici precedono GMMN**, e il VAE a verosimiglianza esplicita è il peggiore;
  - l'HAR con feature generative supera nettamente l'HAR di base nel P&L degli straddle (p.6–7);
  - RCGAN fatica con la bimodalità della deviazione standard su finestra mobile indotta dai regimi (p.7).
- **Nota:** il tracker parla di "superiorità delle GAN". Il quadro corretto è più sfumato, perché i modelli parametrici sono competitivi.

### Masi, Prata, Conti, Bartolini & Vyetrenko (2023) — CoMeTS-GAN
- **Bib:** ACM ICAIF '23 (Sapienza e J.P. Morgan AI Research).
- **Modello:** WGAN condizionale che genera prezzi medi e volumi di più titoli correlati.
- **Dati:** dati LOBSTER intraday (frequenza minuto) di più titoli (es. PEP); più dataset Gaussian e Sines. **Task:** dati sintetici multivariati, simulazione di mercato.
- **Baseline:** TimeGAN e altre.
- **Metriche:** MB (discriminative score); diversità; FB (fatti stilizzati: code, assenza di autocorrelazione, volatility clustering); correlazione di Pearson tra titoli. **SF:** SF1, SF2, SF3, SF9, SF10.
- **Codice:** Y (github.com/giuseppemasi99/COMETS-GAN, p.5).
- **Q:** Q1 P (LOBSTER a pagamento), Q2 Y, Q3 N, Q4 Y, Q5 NA, Q6 Y, Q7 P (più seed).

### Snow (2020) — MTSS-GAN
- **Bib:** SSRN, Alan Turing Institute (preprint, 7 pagine).
- **Modello:** GAN stacked con due encoder, due generatori e due discriminatori.
- **Dati:** Google stock (lo stesso di TimeGAN). **Task:** dati sintetici multivariati.
- **Baseline:** confronto con i numeri riportati da TimeGAN. **Metriche:** MB (predictive score); VIS (t-SNE, PCA); correlazioni tra passi temporali. **SF:** nessuno.
- **Codice:** Y (github.com/firmai/mtss-gan).
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 P, Q5 NA, Q6 N, Q7 N.
- **Nota:** è un lavoro breve e preliminare, con evidenza debole.

### Wang, Hirsa & Li (2024) — MSSTD-Diff
- **Bib:** preprint (Columbia).
- **Modello:** diffusion con RoPE e patching multi-scala, più una loss multi-task (KL, autocorrelazione nel dominio FFT, correlazione, signature) bilanciata con GradNorm; classifier-free guidance per regime.
- **Dati:** rendimenti giornalieri di indici finanziari benchmark. **Task:** dati sintetici multivariati.
- **Baseline:** DDPM con U-Net, FinDiff.
- **Metriche:** FB (momenti, inclusa la curtosi); DB (DTW tra le ACF; distanza Riemanniana tra matrici di correlazione; KL sulle marginali); Explainability Index. **SF:** SF1, SF3, SF9.
- **Codice:** da verificare.
- **Q:** Q1 Y, Q2 ?, Q3 N, Q4 Y, Q5 NA, Q6 Y, Q7 N.
- **Affermazione verificata:** c'è un trade-off tra la fedeltà alle marginali e la fedeltà alla struttura di dipendenza (p.2, p.9–11).

### Li, Meng, Bi, Urnes & Chen (2025) — PaD-TS
- **Bib:** AAAI-25 (p.1).
- **Modello:** diffusion con encoder Transformer a due canali (temporale e cross-dimensionale) e loss MMD a livello di popolazione.
- **Dati:** Sines, Stocks (Google), Energy. **Task:** dati sintetici.
- **Baseline:** Diffusion-TS, TimeGAN, TimeVAE.
- **Metriche:** VDS e FDDS (nuove, a livello di popolazione); MB (DA score e predictive score); le metriche di TSGBench (MDD, ACD, SD, KD, ED, DTW). **SF:** SF9 (dipendenze funzionali).
- **Codice:** Y (github.com/wmd3i/PaD-TS).
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 Y, Q5 NA, Q6 P, Q7 Y.
- **Affermazioni verificate (Tab. 1, p.6):**
  - su Stocks: VDS 0,0029 (PaD-TS) contro 0,0369 (Diffusion-TS); FDDS 0,0588 contro 0,1841;
  - **il predictive score su Stocks non discrimina:** 0,037 per PaD-TS e Diffusion-TS, 0,039 per TimeGAN, 0,038 per TimeVAE;
  - gli autori dichiarano "the unstable nature of the DA score and predictive score" (p.7). È un'evidenza diretta per la Sez. 6.3.

### Chen, Xu, Xu & Zhang (2025/2026) — Diffusion Factor Models
- **Bib:** arXiv 2504.06566v5 (gennaio 2026; preprint).
- **Modello:** diffusion in cui lo score è scomposto tramite proiezioni ortogonali variabili nel tempo (sottospazio dei fattori più componente idiosincratica).
- **Teoria:** limiti non asintotici di errore, per la stima dello score O~(d^{5/2} n^{-2/(k+5)}) e per la distribuzione generata O~(d^{5/4} n^{-1/(2(k+5))}), guidati dalla dimensione dei fattori k (p.1).
- **Dati:** rendimenti in eccesso giornalieri dei titoli USA, maggio 2001 – aprile 2024 (p.27). **Task:** dati sintetici ad alta dimensione e costruzione di portafogli.
- **Baseline:** media e covarianza campionarie; shrinkage; equal weight.
- **Metriche:** stima di media e covarianza; recupero del sottospazio latente; DEC (Sharpe, equivalente certo, turnover, al netto dei costi di transazione). **SF:** SF9.
- **Codice:** Y (github.com/xymmmm00/diffusion_factor_model).
- **Q:** Q1 P (CRSP), Q2 Y, Q3 Y, Q4 N, Q5 Y (OOS trimestrale), Q6 P, Q7 P.
- **Affermazioni verificate:**
  - il portafoglio mean-variance costruito sui dati generati ha circa il doppio dello Sharpe dell'equal weight (p.29);
  - i portafogli tangenti sui fattori ottengono 3–5 volte lo Sharpe dei metodi classici (p.31–32);
  - limite dichiarato: le garanzie teoriche con rumore a code pesanti restano da sviluppare (p.33).

---

## Cartella 04 — Generatori basati su signature, trasporto ottimo e SDE

### Bühler, Horvath, Lyons, Perez Arribas & Wood (2020a/b) — Market Generator con VAE sulle signature
- **Bib (2020a):** arXiv 2006.14498 / SSRN, 27 pagine. **Va citata la versione pubblicata** nella serie della *Conference on Modern Topics in Stochastic Analysis and Applications (in honour of Terry Lyons' 70th birthday)* (NOTE.txt).
- **Bib (2020b):** "Generating Financial Markets with Signatures", 9 pagine. Venue da verificare (credo Risk.net).
- **⚠ PRISMA — duplicato:** le due versioni condividono modello, esperimenti e conclusioni; la 2020b è una versione breve della 2020a. **Propongo di contarle come 1 studio con 2 report.** Il corpus scenderebbe così a **134 studi** (135 report).
- **Modello:** VAE condizionale che lavora sulle log-signature troncate dei segmenti di traiettoria.
- **Dati:** S&P 500 giornaliero (circa 5.000 rendimenti giornalieri, circa 1.000 segmenti settimanali), più esperimenti sintetici. **Task:** dati sintetici in regime di small data, in vista del deep hedging.
- **Baseline:** lo stesso VAE addestrato sui rendimenti invece che sulle signature.
- **Metriche:** PATH (test a due campioni MMD sulle signature, che caratterizza l'intera legge del processo). **SF:** SF1, SF2, SF3, SF5, discussi in rassegna.
- **Codice:** Y (GitHub "Marketsimulator").
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 P, Q5 NA, Q6 P, Q7 N.
- **Affermazioni verificate:**
  - nel 2020a il modello basato su signature funziona "just as well but not significantly better" quando si aumentano i dati di training (p.22);
  - nel 2020b il setup basato sui rendimenti migliora molto con più dati, mentre la versione con signature è affidabile anche con pochi dati (p.9);
  - la Sig-MMD "completely characterizes the law of a stochastic process" (2020b, p.9).

### Ni, Szpruch, Sabate-Vidales, Xiao, Wiese & Liao (2021) — SigWGAN
- **Bib:** ACM ICAIF '21.
- **Modello:** generatore Logsig-RNN (ispirato alle SDE) addestrato con la metrica Sig-W1. Il problema minimax diventa un apprendimento supervisionato.
- **Dati:** GBM multidimensionali; modello di volatilità rough; S&P 500 e DJI giornalieri (p.4–7). **Task:** dati sintetici.
- **Baseline:** WGAN; generatore LSTM. **Metriche:** PATH (Sig-W1); FB (marginali, ACF, correlazioni). **SF:** SF3, SF9.
- **Codice:** Y (github.com/SigCGANs/Sig-Wasserstein-GANs).
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 Y, Q5 NA, Q6 P, Q7 N.

### Liao, Ni, Sabate-Vidales, Szpruch, Wiese & Xiao (2024) — SigCWGAN (condizionale)
- **Bib:** *Mathematical Finance*, DOI 10.1111/mafi.12423 (accettato il 9 ottobre 2023).
- **⚠ CORREZIONE rispetto al tracker:** il **primo autore è Liao**, non Ni, e la sede è *Mathematical Finance*.
- **Modello:** GAN condizionale in cui il W1 condizionale è approssimato nello spazio delle signature; il generatore è un AR-FNN.
- **Dati:** VAR(1) sintetico; ARCH in appendice; SPX/DJI; BTC-USD. **Task:** generazione condizionale (forecasting distribuzionale).
- **Baseline:** CWGAN, TimeGAN, RCGAN, GMMN (p.19).
- **Metriche:** FB (distanza tra le epdf marginali; ACF score al lag 1; correlation score); MB (TSTR con R²); PATH (Sig-W1). **SF:** SF2, SF3, SF9.
- **Codice:** Y.
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 Y, Q5 Y, Q6 P, Q7 P.

### Biagini, Gonon & Walter (2024) — randomised signatures
- **Bib:** arXiv 2406.10214v2 (preprint).
- **Modello:** generatore Neural SDE in stile reservoir, con random feature e solo l'ultimo strato addestrato; discriminatore dato dalla metrica RS-W1 basata sulle randomised signature.
- **Dati:** Brownian motion e AR(1) sintetici; S&P 500; EUR/USD. **Task:** dati sintetici.
- **Baseline:** generatore LSTM; discriminatore Sig-W1.
- **Metriche:** errore sulla covarianza; errore sull'ACF; test di Shapiro-Wilk sulle marginali; il tutto **fuori campione**. **SF:** SF2.
- **Codice:** Y (github.com/niklaswalter/Randomised-Signature-TimeSeries-Generation).
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 Y, Q5 Y, Q6 P, Q7 N.
- **Affermazione verificata:** la coppia Neural SDE + RS-W1 ottiene le metriche fuori campione più basse (Tab. 1, p.15–16).
- **Nota:** la valutazione è centrata su covarianza, ACF e normalità delle marginali, **non su code o volatility clustering**.

### Henry-Labordère (2019) — generatori basati su trasporto ottimo
- **Bib:** SSRN (preprint).
- **Modello:** generatore neurale addestrato minimizzando direttamente la distanza Wp, calcolata con un algoritmo primale-duale e senza discriminatore.
- **Dati:** parametri SABR calibrati giornalmente sulle smile a un anno dell'Eurostoxx (6 dimensioni, 4 anni) (p.5). **Task:** dati sintetici e anomaly detection.
- **Baseline:** GAN, WGAN. **Metriche:** DB (Wp); efficienza della calibrazione. **SF:** nessuno.
- **Codice:** N.
- **Q:** Q1 N, Q2 N, Q3 N, Q4 Y, Q5 NA, Q6 N, Q7 N.
- **Nota:** genera **parametri di un modello di volatilità** (SABR), non serie di rendimenti. È un caso di "generatore di parametri", da collegare a Ning et al. e Cont et al. 2022.

### Remlinger, Mikael & Elie (2022) — CEGEN / Deep Euler
- **Bib:** AAAI-22.
- **⚠ CORREZIONE rispetto al tracker:** l'anno è il 2022 (AAAI), non il 2021.
- **Modello:** generatori con struttura di Eulero:
  - EWGAN ed EDGAN, varianti avversarie;
  - CEGEN, non avversario, che minimizza una distanza di Bures-Wasserstein condizionale tra le leggi di transizione. È dimostrata la convergenza di drift e volatilità per processi di Itô.
- **Dati:** processi simulati (OU e altri, in dimensione crescente); dataset reali di varia natura; esperimento di transfer learning. **Task:** dati sintetici con pochi dati reali.
- **Baseline:** bootstrap (riportato in appendice per i risultati scarsi), TimeGAN, COT-GAN, RCGAN, GMMN.
- **Metriche:** distanza di Fréchet; MSE tra le correlazioni empiriche; stima dei parametri; MB (discriminative e predictive score). **SF:** SF9, SF11 (OU).
- **Codice:** N.
- **Q:** Q1 P, Q2 N, Q3 Y (bootstrap), Q4 Y, Q5 NA, Q6 P, Q7 P.
- **Affermazioni verificate:**
  - CEGEN supera le GAN su dati sintetici e reali (p.2 e p.7);
  - il transfer learning consente di partire da simulazioni Monte Carlo e adattarsi con pochi dati reali (p.1 e p.6).

---

## Cartella 05 — Controllabilità e data augmentation

### Xia, Sun, Wang & An (2024) — Market-GAN
- **Bib:** AAAI-24.
- **Modello:** GAN condizionale con autoencoder e supervisore, sullo schema di TimeGAN, condizionata da un contesto semantico (ticker, stato storico, dinamica di mercato etichettata come bear, flat o bull). Usa C-TimesBlock.
- **Dati:** OHLC giornalieri di 29 titoli del DJIA, 2000–2023. **Task:** dati sintetici controllabili e augmentation.
- **Baseline:** TimeGAN e altri modelli condizionali.
- **Metriche:** una metrica di fedeltà ispirata alla FID; allineamento al contesto; TSTR e augmentation con predittori TimesNet; VIS. **SF:** nessuno.
- **Codice:** da verificare.
- **Q:** Q1 Y, Q2 ?, Q3 N, Q4 Y, Q5 P, Q6 N, Q7 P.

### Tanaka, Hashimoto, Takayanagi, Piao, Murayama & Izumi (2025) — CoFinDiff
- **Bib:** arXiv 2503.04164 (marzo 2025). La sede è da verificare (possibile IJCAI 2025).
- **Modello:** DDPM condizionale su immagini ottenute con trasformata wavelet di Haar, con cross-attention sui target di trend e volatilità realizzata (RV).
- **Dati:** rendimenti a 1 minuto di titoli giapponesi (FLEX full; es. 7203 Toyota). **Task:** dati sintetici controllabili e deep hedging.
- **Baseline:** GBM, GARCH(1,1), GAN vanilla, WGAN (condizionali).
- **Metriche:** FB (curtosi di Fisher, indice di Hill, autocorrelazione); aderenza alle condizioni (errore su trend e RV); DEC (performance del deep hedging). **SF:** SF1, SF3.
- **Codice:** da verificare.
- **Q:** Q1 N (dati proprietari FLEX), Q2 ?, Q3 Y, Q4 Y, Q5 P, Q6 Y, Q7 Y (deviazioni standard).
- **Affermazioni verificate:**
  - curtosi (Tab. p.3): reale 6,97; GBM 0,00; GARCH(1,1) 0,18; GAN 5,39; WGAN 4,27; CoFinDiff 5,16. Un GARCH(1,1) gaussiano su rendimenti a 1 minuto non riproduce le code;
  - errore sulla RV condizionata (p.4): CoFinDiff 5,94 contro 9,99 (GAN) e 16,21 (WGAN).

### Da Silva & Shi (2019) — style transfer su serie storiche
- **Bib:** arXiv 1906.03232v2 (preprint).
- **Modello:** denoising autoencoder con CNN 1D addestrato su dati ad alta frequenza, più style transfer delle statistiche giornaliere. Gli autori motivano esplicitamente la scelta di non usare GAN (instabilità) e VAE (output "sfocati").
- **Dati:** AUD/USD bid, dal 1° maggio 2009 al 31 dicembre 2018 (p.1–2). **Task:** augmentation.
- **Baseline:** GBM, GARCH(1,1) e altri processi stocastici; traiettorie storiche.
- **Metriche:** test del variance ratio di Lo–MacKinlay; autocorrelazione. **SF:** SF2 (struttura non casuale locale).
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 Y, Q4 N, Q5 NA, Q6 P, Q7 P.
- **Affermazione verificata:** i dati storici e quelli generati rifiutano l'ipotesi di random walk, mentre i tre processi stocastici non la rifiutano (p.2).

### Guo (2025) — augmentation con GAN-LSTM
- **Bib:** arXiv (febbraio 2025, 6 pagine; preprint).
- **Modello:** GAN con generatore e discriminatore LSTM.
- **Dati:** prezzi giornalieri da Yahoo (titoli non specificati in dettaglio). **Task:** augmentation per un forecaster LSTM.
- **Baseline:** LSTM addestrato solo sui dati reali. **Metriche:** MSE e R²; ACF tra reale e sintetico. **SF:** SF2 e SF3 in forma qualitativa.
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 N, Q5 ?, Q6 P, Q7 N.
- **Affermazione verificata:** MSE da 0,05 a 0,02 (p.4).
- **Nota:** l'evidenza è molto debole; nella rubrica di qualità questo studio sarà tra i più bassi.

### Aria, Iranmanesh & Hassani (2024) — augmentation con WGAN-GP per il forecasting multivariato
- **Bib:** Journal of Risk and Financial Management 17, 485 (2024).
- **Modello:** WGAN-GP con generatore conv/FC e critic LSTM/FC; forecaster Bi-LSTM.
- **Dati:** indicatori economici rilevanti per il mercato iraniano (S&P 500, Bitcoin, Ethereum, alluminio, oro e altri). **Task:** augmentation per il forecasting multivariato.
- **Baseline:** modelli senza augmentation. **Metriche:** RMSE, MAE, MAPE. **SF:** nessuno.
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 P, Q5 ?, Q6 N, Q7 N.

### Lu (2024) — MA-LSTM
- **Bib:** DEBAI 2024 (ACM, Guangzhou).
- **Modello:** LSTM con attention multi-livello; la GAN serve per l'augmentation.
- **Dati:** azioni A-share cinesi GCEG (Guangdong Construction Engineering Group) e ZTE, per 4.233 giorni, con split 90/10 (p.4). **Task:** augmentation per il forecasting.
- **Baseline:** LSTM, CNN-LSTM, SVM. **Metriche:** MSE e RMSE. **SF:** nessuno.
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 Y, Q5 Y, Q6 N, Q7 N.
- **CORREZIONE rispetto al tracker:** i titoli sono GCEG e ZTE, non Ping An Bank e Vanke.

### Dannels (2023) — recessioni sintetiche con DoppelGANger
- **Bib:** arXiv 2302.10490 (preprint).
- **Modello:** DoppelGANger, con loss di Wasserstein e un discriminatore ausiliario sui metadati.
- **Dati:** rendimenti dei Treasury USA a 1 e 10 anni, giornalieri (FRED); indicatore NBER di recessione. **Task:** augmentation per forecasting e classificazione di eventi rari.
- **Baseline:** modelli addestrati sui soli dati reali (regressione logistica, LSTM). **Metriche:** errore di forecasting; AUC di classificazione; ACF. **SF:** SF4 (persistenza dell'autocorrelazione), SF11.
- **Codice:** N.
- **Q:** Q1 Y, Q2 N, Q3 P, Q4 N, Q5 Y, Q6 P, Q7 N.
- **Affermazioni verificate:**
  - **i dati sintetici faticano a catturare l'autocorrelazione di lungo raggio** delle curve dei rendimenti, perché i campioni di 125 giorni la sottostimano (p.6, p.9, p.12). È un'altra evidenza sulla LRD;
  - nonostante questo, il forecasting di breve periodo e la classificazione delle recessioni migliorano (abstract e p.11).

### Carvajal-Patiño & Ramos-Pollán (2022) — dati sintetici per strategie di trading
- **Bib:** Research in International Business and Finance 62, 101747 (2022).
- **Modello:** confronto tra GAN e VAE; il generatore selezionato è una GAN densa, scelta in base alla distanza di Wasserstein. Il predittore è una random forest.
- **Dati:** XAU/USD ed EUR/USD campionati ogni **5 secondi**. **Task:** augmentation per il trading.
- **Baseline:** modelli addestrati sui soli dati reali. **Metriche:** DB (Wasserstein, per la selezione); accuratezza (circa 59%); DEC (simulazioni di trading e profitto). **SF:** nessuno.
- **Codice:** N.
- **Q:** Q1 P, Q2 N, Q3 N, Q4 P, Q5 Y (attenzione esplicita a non usare dati futuri), Q6 N, Q7 N.
- **Affermazione verificata:** miglioramento consistente fino al profitto nelle simulazioni su commodity e FX (abstract).

### Natarajan, Wang, Jiang, Zhang, Ye & Kuang (2024) [R] — Generative-CNN
- **Bib:** ACM ICAIF '24.
- **⚠ CORREZIONE rispetto al tracker:** l'anno è il 2024 (ICAIF '24), non il 2023.
- **Modello:** DCGAN che genera immagini di grafici a candele per aumentare il training set di una CNN di classificazione dei pattern.
- **Task:** augmentation visiva per il riconoscimento di pattern di analisi tecnica. **Metriche:** accuratezza e F1. **SF:** nessuno.
- **Q:** Q1 P, Q2 ?, Q3 N, Q4 P, Q5 ?, Q6 N, Q7 N.

---

## Cartella 16 — Metriche e benchmark

### Bontorno & Hou (2025) — Nested Optimal Transport Distances
- **Bib:** arXiv 2509.06702 (settembre 2025; preprint, 7 pagine).
- **Contenuto:**
  - manca una metrica di consenso per valutare i modelli generativi su serie finanziarie, analoga alla FID per le immagini (p.1);
  - problemi come hedging, optimal stopping, massimizzazione dell'utilità, minimizzazione del rischio dinamico e RL sono Lipschitz-continui rispetto alla distanza di Wasserstein adattata (AW, o nested), ma non rispetto a W o MMD (p.1–2);
  - viene proposto un algoritmo statisticamente consistente e parallelizzabile per calcolare AW₂².
- **Esperimenti:**
  - convergenza di AW₂² ai valori teorici, con i tempi di calcolo;
  - **un controesempio** (p.3–4): un processo X con le stesse marginali di un moto browniano, ma in cui X₁ è perfettamente prevedibile al tempo δ (look-ahead). W₂ resta piccola mentre AW₂ è grande.
- **Q:** Q1 Y (sintetici), Q2 ?, Q3 NA, Q4 NA, Q5 NA, Q6 NA, Q7 Y (10 run).
- **CORREZIONE rispetto al tracker:** il paper **non** contiene esperimenti che correlano la metrica con il P&L di hedging né confronti tra volatilità rough e standard. Il contributo è teorico-computazionale, con un controesempio sintetico. Va citato come argomento a favore delle metriche causali, non come evidenza empirica sui task a valle.

### Ang, Huang, Bao, Tung & Huang (2023) — TSGBench (metodologico, non primario)
- **Bib:** PVLDB 17(3), 2023 (arXiv 2309.03755v2). Codice: github.com/YihaoAng/TSGBench.
- **Contenuto:**
  - 10 metodi (TimeGAN, TimeVAE, RTSGAN, COSCI-GAN, GT-GAN, AEC-GAN, TimeVQVAE, Fourier Flows, LS4 e altri) su 10 dataset reali, tra cui Stock ed Exchange;
  - 12 misure in 5 categorie: model-based (DS, PS), feature-based (MDD, ACD, SD, KD), distance-based (ED, DTW), visualizzazione (t-SNE, distribuzioni), efficienza;
  - un test di generalizzazione basato sulla domain adaptation;
  - un'analisi statistica dei ranking.
- **Affermazioni verificate:**
  - le pratiche attuali hanno tre limiti (p.1): confronti solo tra modelli dello stesso tipo, dataset privati o sintetici, misure ambigue legate a reti ad hoc;
  - TimeVQVAE, TimeVAE, RTSGAN e COSCI-GAN sono i più consistenti, e i metodi basati su VAE si distinguono (p.8–9);
  - **il discriminative score ha una deviazione standard elevata** (p.8).

---

## Cartella 00 — Background (solo Introduzione e gap)
- **Lee, Guan, Yu & Ding (2024)**, *FinTech* 3, 460–478, DOI 10.3390/fintech3030025. Review che classifica la letteratura con BERTopic; temi dominanti: LLM per la finanza, GAN per i dati sintetici, rischi etici e regolamentari.
- **Assefa, Dervovic, Mahfouz, Tillman, Reddy & Veloso (2020)**, ACM ICAIF '20, DOI 10.1145/3383455.3422554. Opportunità e rischi dei dati sintetici in finanza; manca un "common task framework" per valutarli.
- **Koshiyama, Firoozye & Treleaven (2020)**, ACM ICAIF '20, DOI 10.1145/3383455.3422539. Survey generalista su AI e ML nei capital markets; tratta le GAN in una sola sottosezione.
- **Eckerli & Osterrieder (2021)**, preprint arXiv dell'11 giugno 2021. Overview delle GAN in finanza, con un piccolo esperimento su DCGAN, WGAN-GP e SAGAN.
- **Horvath, Plenk, Saqur & Vuletić (2024)**, preprint (Oxford-Man). Position paper sui "Market Generators" come cambio di paradigma.
- **NoorMohammadzadehMaleki, Abbaskhani, Farhadi & Zamanifar (2025)**, preprint sottomesso a *Computers and Electrical Engineering*. Review sistematica sulla *previsione*, dal machine learning ai Transformer.
- **Shariffudin, Remli & Ni (2025)**, preprint SSRN 5353292, non sottoposto a peer review. Review sulla GenAI per la previsione dei mercati azionari.

---

## Cartella 06 — Forecasting probabilistico e condizionale (tutti nel corpus, canale DB)

### Daiya, Yadav & Rao (2024) — DiffStock
- **Bib:** IEEE ICASSP 2024, DOI 10.1109/ICASSP48485.2024.10446690.
- **Modello:** DDPM condizionato da relazioni tra titoli (ipergrafi), con similarità intra-cluster misurata via DTW.
- **Dati:** StockNet, NASDAQ, NYSE. **Task:** forecasting probabilistico e portafoglio.
- **Baseline:** CSDI, TimeGrad, STHAN-SR, HATS e altre.
- **Metriche:** classificazione (F1, accuratezza, MCC); PROB (CRPS); DEC (Sharpe, strategia buy-hold-sell giornaliera). **SF:** nessuno.
- **Q:** Q1 Y, Q2 ?, Q3 N, Q4 Y, Q5 Y, Q6 N, Q7 ?.

### Istiaque, Pun & Song (2024) — CTS-GAN
- **Bib:** ACM ICAIF '24. **⚠ CORREZIONE:** l'anno è il 2024, non il 2023 come nel tracker.
- **Modello:** estensione condizionale di TimeGAN (cinque reti) che simula prezzi **fuori campione**, un passo avanti, con 10.000 simulazioni.
- **Dati:** titoli dell'S&P 500 per settore, prezzi di chiusura aggiustati giornalieri, 2.623 finestre fuori campione. **Task:** simulazione condizionata e VaR.
- **Baseline:** TimeGAN e altre. **Metriche:** MB (discriminative score); VaR. **SF:** SF1, SF3 (da verificare in dettaglio).
- **Codice:** Y (github.com/riasataliistiaque/cts-gan).
- **Q:** Q1 Y, Q2 Y, Q3 ?, Q4 Y, Q5 Y, Q6 P, Q7 P.
- **Affermazione verificata:** il limite di TimeGAN che motiva il lavoro è che "generates data that is in-sample" (p.1).

### Koa, Ma, Ng & Chua (2023) — D-Va
- **Bib:** CIKM 2023.
- **Modello:** VAE gerarchico accoppiato a un processo di diffusion su input e target, con denoising score matching.
- **Dati:** dataset azionari multipli. **Task:** regressione multi-step dei prezzi e portafoglio mean-variance.
- **Baseline:** ARIMA, Autoformer, NBA e modelli seq2seq.
- **Metriche:** MSE e sua deviazione standard; DEC (Sharpe). **SF:** nessuno.
- **Codice:** Y (github.com/koa-fin/dva).
- **Q:** Q1 Y, Q2 Y, Q3 Y (ARIMA), Q4 Y, Q5 Y, Q6 N, Q7 Y.

### Gao, He, He & Zha (2025/2026) — Factordiff
- **Bib:** arXiv 2509.22088v3 (2 giugno 2026). **⚠ CORREZIONE:** gli autori sono quattro (manca Zha nel tracker).
- **Modello:** Diffusion Transformer con condizionamento token-wise sui fattori specifici di ogni titolo.
- **Teoria:** limite d'errore in W2 sulla performance del portafoglio.
- **Dati:** 118 titoli del CSI 300, giornalieri. **Task:** distribuzione condizionata dei rendimenti per portafogli giornalieri mean-variance e mean-CVaR, con costi di transazione.
- **Baseline:** stimatori empirici, shrinkage, DCC-GARCH. **Metriche:** DEC (Sharpe, Sortino e altre); RISK (CVaR). **SF:** SF9.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 ?, Q5 Y, Q6 N, Q7 ?.

### Kunehepon (2024/2025) — FinGAN++
- **Bib:** preprint (sede da verificare).
- **Modello:** GAN condizionale con generatore LSTM e discriminatore BiLSTM, condizionato dal sentiment estratto con FinBERT.
- **Dati:** titoli ed ETF (es. SPY), giornalieri. **Task:** forecasting condizionato al sentiment e simulazione di scenari.
- **Baseline:** LSTM; modelli basati solo sul sentiment.
- **Metriche:** DEC (rendimento in eccesso, Sharpe, hit rate); robustezza a perturbazioni del sentiment. **SF:** nessuno.
- **Q:** Q1 P, Q2 N, Q3 N, Q4 P, Q5 Y, Q6 N, Q7 N.

### Nie, Mulvey, Poor, Yu & Huang (2026) — Mixture-VAE per l'identificazione dei regimi
- **Bib:** **Annals of Operations Research**, DOI 10.1007/s10479-026-07041-z (accettato il 7/1/2026). **⚠ AGGIORNAMENTO:** nel tracker figurava come preprint del 2025.
- **Modello:** VAE con prior a mistura e transizioni LSTM. La state loss è equivalente alla jump penalty dei jump model.
- **Dati:** dati sintetici; S&P 500 intraday. **Task:** identificazione dei regimi.
- **Baseline:** HMM, statistical jump model e modelli deep. **Metriche:** accuratezza dei regimi; DEC. **SF:** SF11.
- **Codice:** Y (github.com/yuqinie98/Mixture-VAE).
- **Q:** Q1 Y, Q2 Y, Q3 Y, Q4 Y, Q5 Y, Q6 P, Q7 ?.

### Schwarz (2024) — Probabilistic LSTM interpretabile
- **Bib:** il PDF è arXiv (8/7/2024). **Va citata la versione su *The Journal of FinTech*** (NOTE.txt).
- **Modello:** LSTM probabilistica che contiene ARMA-GARCH come caso particolare; un "regolatore" controlla l'interpretabilità.
- **Task:** dati sintetici e forecasting.
- **Baseline:** ARMA-GARCH (statsmodels e arch). **Metriche:** verosimiglianza; distribuzioni. **SF:** SF1, SF3.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 N, Q5 ?, Q6 P, Q7 N.

### Wang, Liu & Zhu (2025) — DiffVT
- **Bib:** Applied Soft Computing 184 (2025) 113763.
- **Modello:** diffusion basata su SDE con un Volatility Transformer ("de-stationary attention") e un campionatore predictor-corrector.
- **Dati:** indici azionari globali. **Task:** forecasting puntuale e probabilistico, trading.
- **Baseline:** TimeGrad, ScoreGrad, Autoformer e altre. **Metriche:** MAE; PROB (CRPS); DEC (Sharpe). **SF:** nessuno.
- **Codice:** Y (github.com/SoraKsgn/DiffVT).
- **Q:** Q1 Y, Q2 Y, Q3 N, Q4 Y, Q5 Y, Q6 N, Q7 ?.
- **Affermazioni verificate:** CRPS −17,23% e Sharpe +12,93% rispetto al secondo miglior modello (abstract).

### Yin, Saef & Barucca (2022) — VHVM
- **Bib:** arXiv 2204.05806. **⚠ CORREZIONE:** gli autori sono tre (manca Saef nel tracker).
- **Modello:** VAE con GRU, che stima la decomposizione di Cholesky della matrice di precisione così che la covarianza resti definita positiva.
- **Dati:** portafogli FX da 5 a 50 asset. **Task:** forecasting della covarianza a un passo.
- **Baseline:** GARCH/DCC, MCMC-SV, Neural Stochastic Volatility Model. **Metriche:** PROB (log-verosimiglianza cumulata). **SF:** SF3, SF9.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 P, Q7 ?.

---

## Cartella 07 — Forecasting avversariale puntuale e direzionale (corpus)

### Kumar, Alsadoon, Prasad, Abdullah, Rashid, Pham & Nguyen (2022) — GAN con PSR ed ERMSE
- **Bib:** Multimedia Tools and Applications 81:3995–4013, DOI 10.1007/s11042-021-11670-w. **Canale:** DB.
- **Modello:** GAN con generatore LSTM e discriminatore CNN 1D, più phase-space reconstruction (PSR).
- **Dati:** DJIA, S&P 500, MSFT, AMZN, GOOG. **Task:** previsione della direzione dei prezzi.
- **Baseline:** LSTM. **Metriche:** accuratezza direzionale, RMSE, tempo di calcolo. **SF:** nessuno.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 P, Q5 ?, Q6 N, Q7 N.
- **Affermazione verificata:** S&P 500, accuratezza dal 58,07% al 63,87% (p.14).

### Lee & Kim (2025) — indice di sentiment FOMC
- **Bib:** *Communications for Statistical Applications and Methods* 32(5):631–654, DOI 10.29220/CSAM.2025.32.5.631. **⚠ Autori:** Kee-Jae Lee e Jong-Min Kim, che nel tracker mancavano. **Canale:** DB.
- **Modello:** GAN densa, confrontata con ARIMA, ETS, MLP, LSTM e CNN.
- **Dati:** indice di sentiment costruito con DistilBERT sugli statement FOMC 1936–2024. **Task:** forecasting.
- **Metriche:** RMSE, MAE, MAPE, con 12 periodi tenuti fuori campione. **SF:** nessuno.
- **Q:** Q1 Y, Q2 Y (codice R/Python), Q3 Y, Q4 P, Q5 Y, Q6 N, Q7 N.
- **Affermazione verificata:** la CNN è la migliore in quasi tutte le metriche, mentre la GAN ha il miglior MAPE sull'indice (p.16).

### Li & Xu (2025) — VAE-GAN-Transformer con FinBERT
- **Bib:** Empirical Economics 68:373–403, DOI 10.1007/s00181-024-02644-6. **Canale:** DB.
- **Modello:** VAE per il denoising, GAN con attention Transformer, sentiment estratto con FinBERT.
- **Dati:** titolo Apple. **Task:** forecasting e trading.
- **Metriche:** RMSE standardizzato; DEC (ROI, Sharpe, Treynor). **SF:** nessuno.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 P, Q5 Y, Q6 N, Q7 N.

### Liu, Wei & Quan (2024) — WGAN-GP con GRU e ResNet
- **Bib:** ACM DEBAI 2024. **Canale:** DB.
- **Modello:** WGAN-GP con generatore GRU e discriminatore CNN-ResNet; usa come proxy di sentiment il bull/bear ratio.
- **Task:** previsione multi-step dei prezzi. **Baseline:** LSTM, GRU, GAN. **Metriche:** RMSE. **SF:** nessuno.
- **Q:** Q1 P, Q2 N, Q3 N, Q4 Y, Q5 ?, Q6 N, Q7 N.

### Shahabi Nejad & Ebadzadeh (2024) — DRAGAN con feature matching
- **Bib:** Expert Systems with Applications 244 (2024) 122952. **⚠ Autori:** individuati sul PDF. **Canale:** DB (SSRN).
- **Modello:** DRAGAN con generatore GRU a due strati, discriminatore CNN a tre strati e feature matching.
- **Dati:** 12 titoli giornalieri 2010–2020, con indicatori tecnici. **Task:** forecasting.
- **Baseline:** LSTM, GAN, WGAN-GP. **Metriche:** RMSE, MAE, R². **SF:** nessuno.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 Y, Q5 Y, Q6 N, Q7 ?.

### Peng, Iima & Kitamura (2025) — FX con eventi di limit order
- **Bib:** Finance Research Letters 72 (2025) 106527. **Canale:** DB.
- **Modello:** GAN con generatore LSTM; input costituito dalle variabili sui limit order e da variabili intraday.
- **Dati:** **sette coppie valutarie**, rendimenti a 5 minuti. **Task:** forecasting e trading.
- **Baseline:** LSTM. **Metriche:** accuratezza; DEC (profitti). **SF:** SF10 (stagionalità intraday come variabile).
- **Q:** Q1 N, Q2 N, Q3 N, Q4 P, **Q5 N**, Q6 N, Q7 ?.
- **⚠ NOTA METODOLOGICA (p.4):** i dati giornalieri vengono **rimescolati** e l'80% dei giorni rimescolati è usato per il training. Lo split quindi non è temporale, con un possibile **look-ahead tra i giorni**. È un esempio utile per la Sez. 6.4 sulla riproducibilità.

### Vuletić, Prenzel & Cucuringu (2024) — Fin-GAN
- **Bib:** Quantitative Finance 24(2):175–199, DOI 10.1080/14697688.2023.2299466. **Canale:** DB.
- **Modello:** ForGAN (CGAN con LSTM), con una loss "economics-driven" che include termini di PnL, Sharpe e MSE.
- **Dati:** excess return giornalieri di titoli ed ETF (31 asset, 2000–2021). **Task:** forecasting e classificazione del segno.
- **Baseline:** LSTM, ARIMA, ForGAN. **Metriche:** DEC (Sharpe annualizzato); MAE; mode collapse. **SF:** nessuno.
- **Codice:** Y (secondo il log di ricerca).
- **Q:** Q1 Y, Q2 Y, Q3 Y (ARIMA), Q4 Y, Q5 Y, Q6 N, Q7 ?.

### Wu, Wang, Li & Deng (2025) — GAN per i rendimenti delle A-share
- **Bib:** Research in International Business and Finance 75 (2025) 102760. **Canale:** DB.
- **Modello:** GAN con LSTM, self-attention, CNN e FNN.
- **Dati:** rendimenti mensili delle A-share cinesi. **Task:** previsione cross-sectional dei rendimenti e portafoglio long-short.
- **Baseline:** OLS, ENet, NN3, LSTM, FNN. **Metriche:** R² fuori campione, MAE, MSE, RMSE; DEC (Sharpe e rendimenti dei portafogli long-short e long-only, Tab. 7 p.8; **CORREZIONE 11/10/2026:** nel PDF non c'è un alpha rispetto a Fama-French). **SF:** nessuno.
- **Q:** Q1 P, Q2 N, Q3 Y, Q4 P, Q5 Y, Q6 N, Q7 ?.

### Ye & Zheng (2023) — GAN e fattori emotivi [nuovo, canale DB]
- **Bib:** IEEE ICDSI 2023, DOI 10.1109/ICDSI60108.2023.00047.
- **Modello:** GAN con predittore MLP. **Dati:** indice azionario dal luglio 2010 e fattori emotivi. **Task:** forecasting. **Metriche:** RMSE e accuratezza.
- **Q:** Q1 ?, Q2 N, Q3 N, Q4 N, Q5 P, Q6 N, Q7 N.
- **Nota:** il testo del PDF ha problemi di codifica; evidenza debole.

### Zhang, Zhong, Dong, Wang & Wang (2019) — GAN per la previsione azionaria [nuovo, canale AM]
- **Bib:** Procedia Computer Science 147:400–406 (IIKI 2018), DOI 10.1016/j.procs.2019.01.256.
- **Modello:** GAN con generatore LSTM e discriminatore MLP.
- **Dati:** S&P 500, indice di Shanghai e titoli, giornalieri con 7 fattori. **Task:** previsione del prezzo di chiusura.
- **Baseline:** LSTM, ANN, SVR. **Metriche:** RMSE, MAE e altre.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 P, Q5 Y, Q6 N, Q7 N.

### Wang & Chen (2024) — Factor-GAN [nuovo, canale AM]
- **Bib:** PLOS ONE 19(6):e0306094, DOI 10.1371/journal.pone.0306094.
- **Modello:** GAN con generatore LSTM e discriminatore CNN, su 70 caratteristiche d'impresa.
- **Dati:** tutte le A-share cinesi, 2002–2020. **Task:** previsione dei rendimenti e factor investing.
- **Baseline:** modelli lineari (FF3, FF5) e ML. **Metriche:** R² fuori campione; DEC (portafoglio long-short con rendimento annualizzato del 23,52% e Sharpe 1,29).
- **Q:** Q1 P, Q2 N, Q3 Y, Q4 P, Q5 Y, Q6 N, Q7 ?.

---

## Cartella 08 — Componenti generativi come estrattori di rappresentazioni o denoiser (corpus, categoria di confine)

### Arabyat, AlZubi, Aldebei & Al-oqaily (2022)
- **Bib:** Risks 10(8):151, DOI 10.3390/risks10080151. **Canale:** DB.
- **Modello:** autoencoder profondo ottenuto impilando RBM, seguito da una rete BP per la previsione.
- **Dati:** titolo Kweichow Moutai. **Baseline:** PCA, analisi fattoriale. **Metriche:** errore di previsione.
- **Q:** Q1 P, Q2 N, Q3 P, Q4 N, Q5 ?, Q6 N, Q7 N.

### Chiang & Chi (2024) — CycleGAN su immagini GADF
- **Bib:** FinTech 3:427–459, DOI 10.3390/fintech3030024. **Canale:** DB.
- **Modello:** CycleGAN (generatore ResNet, discriminatore CNN) che mette in relazione prezzo e volume rappresentati come immagini GADF; più system dynamics.
- **Dati:** titolo TSMC. **Task:** previsione e trading con bande di Bollinger. **Metriche:** ROI, Sharpe, VaR.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 P, Q5 ?, Q6 N, Q7 N.

### Hu, Zhang, Liu, Yu, Zheng & Zhou (2024) — Autoencoder-CNN-GAN per il trading cripto
- **Bib:** IEEE ICAIRC 2024, DOI 10.1109/ICAIRC64177.2024.10899902. **Canale:** DB.
- **Modello:** autoencoder per il denoising, CNN e GAN usata come estrattore di feature (vettore a 160 dimensioni).
- **Dati:** BTC perpetual, ottobre 2019 – ottobre 2024. **Baseline:** ARIMA, LSTM, CNN-LSTM. **Metriche:** accuratezza (61,2%); DEC (Sharpe 2,5; MDD 15%).
- **Q:** Q1 P, Q2 N, Q3 Y, Q4 P, Q5 Y, Q6 N, Q7 N.

### Sarkar & Vadivu (2025) — ensemble VAE + Transformer + LSTM
- **Bib:** sede da verificare (probabilmente un atto di conferenza). **Canale:** DB (arXiv).
- **Modello:** il VAE estrae le feature, combinate in ensemble con Transformer e LSTM.
- **Dati:** prezzi giornalieri da yfinance con indicatori tecnici. **Baseline:** modelli singoli. **Metriche:** errore e accuratezza direzionale.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 N, Q5 ?, Q6 N, Q7 N.

### Troiano, Mejuto & Kriplani (2017) [R] — riduzione delle feature con RBM e autoencoder
- **Bib:** arXiv 1704.03205. **Canale:** DB (arXiv).
- **Modello:** RBM e autoencoder per ridurre la dimensione degli indicatori tecnici, poi un classificatore del trend.
- **Dati:** S&P 500. **Metriche:** accuratezza di classificazione.
- **Q:** Q1 Y, Q2 N, Q3 N, Q4 N, Q5 ?, Q6 N, Q7 N.

### Wang & Ventre (2024) — diffusion come denoiser
- **Bib:** ACM ICAIF '24, DOI 10.1145/3677052.3698649. **Canale:** DB.
- **Modello:** diffusion con SDE VE/VP e Transformer; noising e denoising parziali con classifier-free guidance.
- **Dati:** titoli dell'S&P 500 (fonte Stooq), giornalieri e a 5 minuti. **Task:** denoising per forecasting e trading (MACD, Bollinger).
- **Baseline:** EMA, DAE. **Metriche:** prevedibilità; numero di eventi "directional change"; DEC. **SF:** nessuno.
- **Q:** Q1 Y, Q2 ?, Q3 Y (EMA), Q4 Y (DAE), Q5 Y, Q6 N, Q7 Y (5 seed).

---

## Cartella 09 — Microstruttura e LOB (corpus)

### Hultin, Hult, Proutiere, Samama & Tarighati (2023)
- **Bib:** Quantitative Finance 23(6):931–958, DOI 10.1080/14697688.2023.2205583. **Canale:** DB.
- **Modello:** generatore autoregressivo con una RNN per ciascuna componente dell'evento (tipo, prezzo, quantità, tempo).
- **Dati:** Nasdaq Stockholm. **Task:** simulazione degli eventi del LOB ed esecuzione.
- **Baseline:** un modello naive. **Metriche:** statistica KS per tipo di evento e livello di prezzo; ACF dei rendimenti a 10 secondi e dei loro quadrati; price signature (impatto di mercato). **SF:** SF2, SF3, SF10.
- **Q:** Q1 N, Q2 ?, Q3 P, Q4 N, Q5 Y, Q6 Y, Q7 ?.

### Lim & Gorse (2021) — SeqGAN sull'order flow
- **Bib:** IEEE ICMLA 2021, DOI 10.1109/ICMLA52953.2021.00068. **Canale:** DB.
- **Modello:** SeqGAN che genera sequenze di ordini; il prezzo emerge dopo il passaggio in un simulatore LOB.
- **Dati:** 5 titoli tecnologici. **Baseline:** un modello benchmark.
- **Metriche:** test KS a due campioni (su 100 ipotesi); curtosi, esponente di coda, test di Jarque-Bera. **SF:** SF1, SF3, SF10.
- **Q:** Q1 P, Q2 N, Q3 Y, Q4 N, Q5 Y, Q6 Y, Q7 P.

### Li, Liu, Liu, Fang, Wang, Xu & Bian (2025) — MarS
- **Bib:** ICLR 2025. **Canale:** DB (arXiv).
- **Modello:** "Large Market Model", Transformer decoder-only su ordini e lotti di ordini.
- **Task:** simulazione interattiva, impatto di mercato, scenari controfattuali. **Baseline:** modelli di impatto noti.
- **Metriche:** fatti stilizzati (aggregational Gaussianity, assenza di autocorrelazione, volatility clustering); scaling laws; impatto di mercato. **SF:** SF2, SF3, SF7, SF10.
- **Codice:** Y (github.com/microsoft/MarS).
- **Q:** Q1 N, Q2 Y, Q3 Y, Q4 ?, Q5 Y, Q6 Y, Q7 ?.

### Li & Chen (2025) — ByteGen
- **Bib:** arXiv 2508.02247v2. **Canale:** DB (arXiv).
- **Modello:** generatore senza tokenizer che lavora sui byte grezzi; architettura H-Net (Mamba più Transformer) con chunking dinamico.
- **Dati:** futures BTC del CME, livello 3. **Metriche:** fatti stilizzati (code pesanti, clustering), distribuzioni di prezzo. **SF:** SF1, SF3, SF10.
- **Q:** Q1 N, Q2 ?, Q3 N, Q4 ?, Q5 Y, Q6 Y, Q7 ?.

### Cont, Cucuringu, Kochems & Prenzel (2023) — LOB con GAN
- **Bib:** SSRN (preprint). **Canale:** DB (SSRN).
- **Modello:** WGAN condizionale su snapshot del LOB, con supporto per livelli vuoti ed effetti intraday.
- **Dati:** LOBSTER (Nasdaq). **Baseline:** VAR lineare. **Metriche:** forma del LOB, autocorrelazioni, impatto di mercato. **SF:** SF3, SF10.
- **Q:** Q1 P, Q2 N, Q3 Y, Q4 N, Q5 Y, Q6 Y, Q7 ?.

### Cont, Cucuringu, Kochems & Prenzel (2022) — calibrazione dinamica dell'order flow
- **Bib:** ACM ICAIF '22. **Canale:** DB.
- **Modello:** GAN che genera le **intensità di Poisson** dell'order flow (stimate su intervalli di 6 minuti), condizionate all'ora del giorno e al VIX.
- **Dati:** un anno di dati LOBSTER. **Baseline:** il modello di Poisson con calibrazione statica. **Metriche:** pattern intraday a U, distribuzione delle intensità. **SF:** SF10.
- **Q:** Q1 P, Q2 N, Q3 Y, Q4 N, Q5 ?, Q6 P, Q7 ?.

### Coletta, Moulin, Vyetrenko & Balch (2022) — "World Agent"
- **Bib:** ACM ICAIF '22. **Canale:** DB.
- **Modello:** WGAN-GP condizionale che genera ordini (anche discreti) come un "world agent" dentro ABIDES.
- **Metriche:** fatti stilizzati (code, autocorrelazione, ACF dei rendimenti al quadrato), reattività. **SF:** SF1, SF2, SF3, SF10.
- **Q:** Q1 P, Q2 ?, Q3 P, Q4 ?, Q5 Y, Q6 Y, Q7 P.

### Coletta, Prata, Conti, Mercanti, Bartolini, Moulin, Vyetrenko & Balch (2021)
- **Bib:** ACM ICAIF '21 (Sapienza e J.P. Morgan AI Research). **Canale:** DB.
- **Modello:** WGAN condizionale integrata in ABIDES, testata con un agente sperimentale POV.
- **Dati:** LOBSTER. **Metriche:** fatti stilizzati (assenza di autocorrelazione, clustering, code pesanti, aggregational normality); impatto di mercato. **SF:** SF1, SF2, SF3, SF7, SF10.
- **Q:** Q1 P, Q2 ?, Q3 P, Q4 ?, Q5 Y, Q6 Y, Q7 P.

### Coletta, Jerome, Savani & Vyetrenko (2023a) — generatori condizionali per ambienti LOB
- **Bib:** ACM ICAIF '23 (da confermare). **Canale:** DB.
- **Modello:** LOBGAN, WGAN condizionale.
- **Contenuto:** explainability; attacchi avversari tramite il book imbalance; versioni v1–v3 più robuste.
- **Dati:** LOBSTER (gennaio 2021). **Metriche:** traiettorie confrontate con una baseline, intervalli di confidenza al 5°–95° percentile, profitto del market making.
- **Q:** Q1 P, Q2 ?, Q3 N, Q4 P, Q5 Y, Q6 P, Q7 Y.

### Berti, Prenkaj & Velardi (2025) — TRADES
- **Bib:** **ECAI 2025**, FAIA, IOS Press, DOI 10.3233/FAIA251249 (Sapienza). **⚠ AGGIORNAMENTO:** è una pubblicazione peer-reviewed, non solo un preprint. **Canale:** AM nel tracker; **va verificato** se è nel Merge DB (risulta tra i record arXiv, quindi canale DB).
- **Modello:** DDPM condizionale con backbone Transformer; DeepMarket, framework open source basato su ABIDES.
- **Dati:** LOBSTER (TSLA, INTC). **Baseline:** CGAN, IABS e altre.
- **Metriche:** predictive score adattato (×3,48 rispetto allo stato dell'arte); fatti stilizzati; impatto di mercato con un agente POV. **SF:** SF2, SF3, SF10.
- **Codice:** Y (DeepMarket).
- **Q:** Q1 P, Q2 Y, Q3 P, Q4 Y, Q5 Y, Q6 Y, Q7 ?.

### Wang & Ventre (2025) — DiffVolume
- **Bib:** ACM ICAIF '25. **Canale:** DB (arXiv).
- **Modello:** diffusion score-based condizionale sui volumi del LOB a 20 livelli, condizionata all'ora del giorno e a una liquidità target.
- **Dati:** LOBSTER, 4 titoli. **Baseline:** GAN (incondizionata e condizionale).
- **Metriche:** DB (W, KL, KS); curtosi; correlazione dei volumi tra livelli; augmentation con LightGBM. **SF:** SF1, SF9 (tra livelli), SF10.
- **Q:** Q1 P, Q2 ?, Q3 N, Q4 Y, Q5 Y, Q6 Y, Q7 ?.

### Backhouse, Li, Foerster, Calinescu & Zohren (2025) — "Painting the market"
- **Bib:** arXiv 2509.05107. **Canale:** DB (arXiv).
- **Modello:** diffusion che tratta il LOB come un'immagine da completare (inpainting).
- **Dati:** LOBSTER; LOB-Bench. **Baseline:** il modello classico di Cont et al., oltre a modelli generativi.
- **Metriche:** perdite L1 e Wasserstein. Il modello fatica su INTC (p.5), che però è un titolo **a tick grande** (p.4; **CORREZIONE 11/10/2026**); su INTC va in mode collapse e LOBS5 è dominante, mentre su GOOG (tick piccolo) è allo stato dell'arte sulla loss di Wasserstein. **SF:** SF10.
- **Q:** Q1 P, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 P, Q7 Y (intervalli di confidenza).

### Takahashi & Mizuno (2025) — DDPM su immagini wavelet
- **Bib:** Quantitative Finance 25(10):1507–1516, DOI 10.1080/14697688.2025.2528697. **Canale:** DB.
- **Modello:** DDPM su immagini ottenute con wavelet di Haar; genera insieme prezzo, spread e volume.
- **Dati:** AAPL, dati al minuto. **Baseline:** TimeGAN, QuantGAN, DDPM senza wavelet.
- **Metriche:** fatti stilizzati (code, clustering, stagionalità a U di spread e volume, correlazioni incrociate). **SF:** SF1, SF3, SF9, SF10.
- **Q:** Q1 N, Q2 ?, Q3 N, Q4 Y, Q5 NA, Q6 Y, Q7 ?.

### Storchan, Vyetrenko & Balch (2021) — calibrazione avversaria di simulatori multi-agente
- **Bib:** arXiv 2108.00664. **Canale:** DB (arXiv).
- **Modello:** un discriminatore GAN con self-attention fa da obiettivo per calibrare la composizione degli agenti in ABIDES.
- **Metriche:** test KS; 20 simulazioni con seed diversi. **SF:** SF10.
- **Q:** Q1 P, Q2 ?, Q3 Y (ABM), Q4 N, Q5 NA, Q6 P, Q7 Y.

### Cetin & Minca (2024) [nuovo, canale DB]
- **Bib:** IEEE CIFEr 2024, DOI 10.1109/CIFER62890.2024.10772805.
- **Modello:** PGSGAN, una GAN ispirata al reinforcement learning, confrontata con un modello Markoviano a tempo continuo (CTMC) simulato via Monte Carlo.
- **Dati:** LOB di Bitcoin su Coinbase (6 ore), più dati simulati. **Metriche:** distribuzione dei prezzi; previsione con LightGBM.
- **Affermazione verificata:** il modello agnostico "is able to match the price distribution of CTMC-based models".
- **Q:** Q1 P, Q2 N, Q3 Y, Q4 N, Q5 ?, Q6 N, Q7 Y (IC al 95%).

### Nagy, Frey, Sapora, Li, Calinescu, Zohren & Foerster (2023) [nuovo, canale DB]
- **Bib:** ACM ICAIF '23, DOI 10.1145/3604237.3626898.
- **Modello:** generatore autoregressivo a token (state-space S5) che produce i messaggi del LOB, interpretati dal simulatore JAX-LOB.
- **Dati:** LOBSTER (GOOG, INTC). **Metriche:** perplexity; distribuzione dei rendimenti del mid-price; correlazioni. **SF:** SF10.
- **Codice:** Y (github.com/peernagy/LOBS5).
- **Q:** Q1 P, Q2 Y, Q3 N, Q4 N, Q5 Y, Q6 P, Q7 ?.

---

## Cartella 10 — Rischio di mercato: VaR ed ES (corpus)

### Allen, Mushunje & Peiris (2025) — GAN, dati sintetici e calcolo del VaR
- **Bib:** Applied Economics 57(37):5680–5695, DOI 10.1080/00036846.2024.2365456. **Canale:** DB (Scopus).
- **⚠ CORREZIONE IMPORTANTE:** il tracker attribuisce questo titolo a "Roque & Cavique 2021". Il record di Scopus e il PDF sono invece di **Allen, Mushunje & Peiris**.
- **Modello:** WGAN. **Dati:** S&P 500 e FTSE 100, giornalieri. **Task:** dati sintetici per il VaR.
- **Metriche:** confronto dei momenti; VaR. **SF:** SF1 (momenti).
- **Q:** Q1 Y, Q2 N, Q3 P (simulazione storica), Q4 N, Q5 ?, Q6 P, Q7 ?.

### Arian, Moghimi, Tabatabaei & Zamani (2020) — Encoded VaR
- **Bib:** arXiv 2011.06742. **⚠ CORREZIONE:** il primo autore è **Arian**, non Moghimi. **Canale:** DB (SSRN).
- **Modello:** VAE che genera scenari cross-sectional dei rendimenti di portafoglio.
- **Dati:** portafogli su S&P 500, LSE e FSE. **Task:** VaR.
- **Baseline:** 11 metodi VaR (CAViaR, E-GARCH, simulazione storica e altri). **Metriche:** backtest di Kupiec e Christoffersen; loss function di Caporin, Sener e Sarma. **SF:** SF1.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 N, Q5 Y, Q6 P, Q7 ?.

### Chen (2024) — FE-GAN
- **Bib:** arXiv 2411.15519 (UCL). **Canale:** DB (arXiv).
- **Modello:** WGAN e Tail-GAN arricchite di feature storiche usate come condizionamento implicito.
- **Task:** VaR ed ES. **Baseline:** simulazione storica, varianza-covarianza. **Metriche:** RISK (errore di backtest su VaR ed ES).
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 N, Q7 ?.

### Cont, Cucuringu, Xu & Zhang (2026) — TAIL-GAN
- **Bib:** **Management Science 72(4):2917–2936 (aprile 2026)**. **⚠ AGGIORNAMENTO:** il tracker lo indica come preprint del 2025. **Canale:** DB (SSRN).
- **Modello:** GAN il cui discriminatore usa score congiuntamente elicitabili per (VaR, ES) calcolati sul P&L di strategie benchmark; è dimostrata l'approssimazione universale.
- **Dati:** sintetici e intraday reali (NASDAQ, LOBSTER). **Baseline:** GAN standard, simulazione storica. **Metriche:** RISK (VaR ed ES di portafogli dinamici). **SF:** SF1.
- **Q:** Q1 P, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 P, Q7 ?.

### Ericson, Zhu, Han, Fu, Li, Guo & Hu (2024) — confronto per il VaR
- **Bib:** arXiv 2401.10370. **Canale:** DB (arXiv).
- **Modelli:** simulazione storica, GARCH, CWGAN, CGAN encoder-decoder, Conditional TimeVAE, diffusion, SigWGAN.
- **Dati:** curva dei rendimenti USD; dati simulati da GARCH e CIR. **Metriche:** distanza tra distribuzioni, autocorrelazione, backtest del VaR. **SF:** SF3.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 P, Q7 ?.
- **Affermazione verificata:** "top performing models are HS, GARCH and CWGAN" (abstract).

### Fu, Chen, Zeng, Zhuang & Sudjianto (2019) — CGAN (Wells Fargo)
- **Bib:** preprint (Corporate Model Risk, Wells Fargo). **Canale:** DB (SSRN).
- **Modello:** CGAN con condizionamenti categoriali e continui.
- **Dati:** simulazioni (misture, code pesanti, dipendenze) e dati reali. **Task:** VaR ed ES.
- **Baseline:** simulazione storica, KDE. **Metriche:** RISK. **SF:** SF1, SF9.
- **Q:** Q1 P, Q2 N, Q3 Y, Q4 N, Q5 Y, Q6 P, Q7 ?.

### Hassan (2025) — dati sintetici per le banche sistemiche (G-SIFI)
- **Bib:** Computational Economics 67:3815–3875, DOI 10.1007/s10614-025-10956-7. **Canale:** DB.
- **Modello:** CGAN condizionata alla volatilità realizzata.
- **Dati:** 39 G-SIFI di USA, UK ed Europa, 2018–2023. **Baseline:** EVT con Peaks over Threshold.
- **Metriche:** RISK (VaR, CVaR); distribuzione; autocorrelazione. **SF:** SF1, SF3.
- **Q:** Q1 Y, Q2 N, Q3 Y, Q4 N, Q5 ?, Q6 Y, Q7 ?.
- **Affermazioni verificate (abstract):** il modello POT è superiore nella stima del rischio di coda, mentre la CGAN sovrastima le metriche di rischio del 15–20%.

### Istiaque, Pun & Yong (2025) — HMGM
- **Bib:** Quantitative Finance 25(6):873–893, DOI 10.1080/14697688.2025.2511115. **Canale:** DB.
- **Modello:** HMM per i regimi più una WGAN per ciascun regime.
- **Dati:** 505 titoli dell'SPX, 1999–2023, fino a 64 asset. **Baseline:** GARCH e modelli parametrici.
- **Metriche:** fatti stilizzati (code, clustering, leverage, correlazione gerarchica); RISK (VaR). **SF:** SF1, SF3, SF5, SF9, SF11.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 ?, Q5 Y, Q6 Y, Q7 ?.

### Kondratyev & Schwarz (2020) — The Market Generator (RBM)
- **Bib:** SSRN (gennaio 2020). **Canale:** DB (SSRN).
- **Modello:** RBM di Bernoulli su dati binarizzati a 16 bit; il numero di passi di Gibbs controlla l'autocorrelazione.
- **Dati:** EURUSD, GBPUSD, USDJPY, USDCAD; 5.070 log-rendimenti giornalieri. **Baseline:** copula gaussiana.
- **Metriche:** marginali, code, correlazioni lineari e di rango, dipendenza di coda. **SF:** SF1, SF9.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 N, Q5 NA, Q6 Y, Q7 N.

### Noorani, Astuti, Bruno, Ergun & Skavysh (2025) — QBM
- **Bib:** working paper (University of Maryland, Banca d'Italia, Bank of Canada; 13/6/2025). **Canale:** DB (SSRN).
- **Modello:** Quantum Boltzmann Machine.
- **Task:** augmentation per stimare il rischio di coda con pochi dati (società appena quotate). **Baseline:** RBM classica, bootstrap.
- **Metriche:** RISK (VaR ed ES di lungo periodo). **SF:** SF1.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 P, Q7 ?.

### Sicks/Buch, Grimm, Korn & Richert — TempVAE
- **Bib:** il PDF è arXiv 2112.01896 (Sicks et al. 2021). **Il record di Scopus è *Risks* 11(5):79 (2023), DOI 10.3390/risks11050079, autori Buch, Grimm, Korn & Richert.** **⚠ Va citata la versione su *Risks*.** Probabilmente si tratta dello stesso primo autore con il cognome cambiato; da verificare. **Canale:** DB.
- **Modello:** VRNN con bottleneck, senza componente autoregressiva, con β-annealing.
- **Dati:** portafogli DAX e S&P 500. **Baseline:** GARCH, DCC-GARCH, simulazione storica.
- **Metriche:** RISK (VaR al 95% e al 99%; regulatory loss). **SF:** nessuno in modo esplicito.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 N, Q5 Y, Q6 N, Q7 ?.

### Tepelyan & Gopal (2023) — CIWAE e flow per i rendimenti azionari
- **Bib:** ACM ICAIF '23 (Bloomberg). **Canale:** DB.
- **Modello:** modello dei fattori con CIWAE, più normalizing flow condizionale per i singoli titoli.
- **Dati:** costituenti dell'S&P 500. **Baseline:** GARCH, GJR-GARCH, modello a fattori lineare.
- **Metriche:** PROB (NLL univariata e multivariata); RISK (VaR); DEC (Sharpe long-short). **SF:** SF1, SF3, SF5, SF9.
- **Q:** Q1 P, Q2 ?, Q3 Y, Q4 N, Q5 Y, Q6 Y, Q7 ?.

### Yang, Gong & Fang (2025) [nuovo, canale DB]
- **Bib:** Computational Economics 67:4169–4197, DOI 10.1007/s10614-025-11004-0.
- **Modello:** GAN combinata con GARCH-VaR per stimare le distribuzioni, poi analisi quantilica condizionata.
- **Dati:** commodity (energia, metalli, agricoltura) e indici di finanza green; 3.675 osservazioni giornaliere, novembre 2013 – dicembre 2023.
- **Task:** spillover di rischio estremo. **Metriche:** RISK (VaR; spillover).
- **Q:** Q1 Y, Q2 N, Q3 Y, Q4 N, Q5 ?, Q6 N, Q7 ?.

---

## Cartella 11 — Scenari, stress test, rischio di credito e rischio sistemico (corpus)

### Caprioli, Cagliero & Crupi (2023) — VAE per le correlazioni del Credit VaR
- **Bib:** arXiv 2309.08652v2 (Intesa Sanpaolo). **Canale:** DB (Scopus).
- **Modello:** VAE che genera matrici di correlazione, con un latente bidimensionale interpretabile.
- **Task:** sensitività del Credit VaR (modello di Vasicek multifattoriale). **Metriche:** fatti stilizzati delle correlazioni (Marchenko–Pastur, Perron–Frobenius, struttura gerarchica); RISK (Credit VaR). **SF:** SF9.
- **Q:** Q1 P, Q2 ?, Q3 Y (stress classici), Q4 N, Q5 NA, Q6 Y, Q7 N.

### Coletta, Gopalakrishnan, Borrajo & Vyetrenko (2023b) — generazione vincolata di serie storiche
- **Bib:** NeurIPS 2023 (J.P. Morgan AI Research). **Canale:** AM (incluso con l'opzione B per il focus finanziario).
- **Metodi:** vincoli soft e hard imposti al momento dell'inferenza; un problema di ottimizzazione vincolata (COP); "guided" DiffTime; confronto con il rejection sampling.
- **Metriche:** rispetto dei vincoli, realismo (autocorrelazione dei rendimenti), tempi di calcolo. **SF:** SF2.
- **Q:** Q1 Y, Q2 ?, Q3 N, Q4 Y, Q5 NA, Q6 P, Q7 ?.

### Feng, Min & Fouque (2022) — deep learning per le misure di rischio sistemico
- **Bib:** ACM ICAIF '22. **Canale:** DB.
- **Metodo:** gioco min-max in stile GAN per risolvere il problema duale della misura di rischio sistemico: le reti stimano la derivata di Radon–Nikodym e l'allocazione ottimale. Rientra nella **categoria di confine** dei generatori di decisioni.
- **Metriche:** errore rispetto alle soluzioni in forma chiusa; scalabilità da 10 a 100 istituzioni.
- **Q:** Q1 Y (simulato), Q2 ?, Q3 Y (soluzioni analitiche), Q4 N, Q5 NA, Q6 NA, Q7 ?.

### Fukunishi, Qiu, Takahashi & Ye (2026) — diffusion per la struttura a termine dei tassi
- **Bib:** **Expert Systems with Applications 328 (2026) 132895.** **⚠ AGGIORNAMENTO:** nel tracker figura come working paper. **Canale:** DB (SSRN).
- **Modello:** DDPM con U-Net 2D, cross-attention e v-parameterization; due modalità: generazione dell'intera curva o dei fattori Nelson–Siegel–Svensson.
- **Dati:** tassi OIS giornalieri in JPY, USD e GBP, interpolati con spline di Hermite; condizionamento su indicatori di QE e base monetaria. **Metriche:** RMSE; generazione fuori campione. **SF:** SF11.
- **Q:** Q1 P, Q2 ?, Q3 Y (NSS), Q4 N, Q5 Y, Q6 P, Q7 ?.

### Genovese, Nikeghbali, Serra & Visentin (2023) — RBM per le perdite di portafogli di credito
- **Bib:** arXiv 2202.11060v3 (25/4/2023). **Canale:** DB (arXiv).
- **Modello:** RBM con importance sampling.
- **Dati:** probabilità di default giornaliere a un anno di 1.012 società USA. **Baseline:** copule fattoriali parametriche (gaussiana e t).
- **Metriche:** DB (MMD); RISK (VaR, stressed VaR sotto scenari macro); bande di confidenza da bootstrap parametrico. **SF:** SF9.
- **Codice:** Y (github.com/gvisen/credit-RBM-model).
- **Q:** Q1 P, Q2 Y, Q3 Y, Q4 N, Q5 Y, Q6 P, Q7 Y.

### Kim, Hong & Lee (2023) — anomaly detection con GAN
- **Bib:** ACM ICAIF '23. **Canale:** DB.
- **Modello:** GAN con inverse mapping; l'anomaly score è l'errore di ricostruzione.
- **Dati:** S&P 500 e 11 indici settoriali. **Task:** early warning e strategia difensiva.
- **Metriche:** DEC (rendimento, volatilità, MDD, Sharpe rispetto al buy & hold dell'S&P 500).
- **Q:** Q1 Y, Q2 N, Q3 P, Q4 N, Q5 Y, Q6 N, Q7 ?.

### Rizzato, Wallart, Geissler, Morizet & Boumlaika (2023) — Jinkou
- **Bib:** Physica A 623 (2023) 128899. **Canale:** DB (SSRN).
- **Modello:** BiGAN per le variabili di stato globali (macro) più una CGAN per le transizioni degli asset da t a t+h.
- **Baseline:** copula gaussiana; viene discusso il bootstrap. **Metriche:** KS, punteggio PCA, accuratezza di classificazione reale/sintetico. **SF:** SF1, SF9.
- **Q:** Q1 P, Q2 ?, Q3 Y, Q4 N, Q5 ?, Q6 P, Q7 ?.

### Rusnak & Daul (2023) — MC-TE-GAN
- **Bib:** **ACM ICAIF '24**, DOI 10.1145/3677052.3698606 (**CORREZIONE 11/10/2026:** non ICAIF '23). Titolo: "Macroeconomic Conditioned Synthetic Financial Markets". **⚠ CORREZIONE:** gli autori sono due (manca Daul, Pictet AM). **Canale:** DB.
- **Modello:** GAN con encoder Transformer, condizionata a variabili macroeconomiche.
- **Dati:** rendimenti azionari giornalieri multivariati. **Baseline:** CoMeTS-GAN.
- **Metriche:** fatti stilizzati (autocorrelazione, curtosi, clustering, leverage); correlazioni. **SF:** SF1, SF2, SF3, SF5, SF9.
- **Q:** Q1 P, Q2 ?, Q3 N, Q4 Y, Q5 ?, Q6 Y, Q7 ?.

### Leangarun, Tangamchit & Thajchayapong (2018) — rilevazione di manipolazioni di prezzo [nuovo, canale AM]
- **Bib:** IEEE (2018); venue da verificare. Link: ieeexplore.ieee.org/document/8628777.
- **Modello:** GAN con LSTM; il discriminatore fa da rilevatore di manipolazioni.
- **Dati:** dati di trading della Stock Exchange of Thailand, con manipolazioni pump-and-dump simulate per il test. **Metriche:** accuratezza (68,1%).
- **Q:** Q1 N, Q2 N, Q3 N, Q4 N, Q5 Y, Q6 N, Q7 N.

### Kubiak, Weyde, Galkin, Philps & Gopal (2025) — MacroVAE [nuovo, canale AM]
- **Bib:** ACM ICAIF '25, DOI 10.1145/3768292.3770360.
- **Modello:** VAE condizionale con blocchi ResNet convoluzionali e modulazione FiLM, condizionato a variabili macroeconomiche.
- **Dati:** rendimenti di futures globali e dati macro. **Task:** scenari controfattuali.
- **Baseline:** benchmark storici e generativi (QuantGAN, FIN-GAN e altri). **Metriche:** distribuzioni e fatti stilizzati, con valutazione fuori campione a finestra mobile.
- **Codice:** Y (github.com/szymkubiak/macrovae).
- **Q:** Q1 Y, Q2 Y, Q3 Y, Q4 Y, Q5 Y, Q6 Y, Q7 ?.

---

## Cartella 12 — Derivati: pricing, hedging, superfici IV (corpus)

### Wiese, Bai, Wood & Buehler (2019) — Deep Hedging: simulare il mercato delle opzioni equity
- **Bib:** arXiv 1911.01700; NeurIPS 2019 (con ogni probabilità il workshop su AI in finanza; da verificare). **Canale:** DB (SSRN).
- **⚠ CORREZIONE IMPORTANTE:** il modello è una **GAN**: architetture ricorrenti e TCN, WGAN-GP, compressione dello stato. **Non è un CVAE con log-signature**, come indica il tracker confondendolo con Bühler et al.
- **Dati:** volatilità implicite dell'EURO STOXX 50. **Baseline:** qMLE (modello classico).
- **Metriche:** punteggi sulle proprietà distributive delle volatilità implicite (benchmark scores). **SF:** SF3, SF5.
- **Q:** Q1 N, Q2 ?, Q3 Y, Q4 Y, Q5 ?, Q6 Y, Q7 ?.

### Acciaio, Eckstein & Hou (2024) — Time-Causal VAE
- **Bib:** working paper ETH (5/11/2024), DOI 10.3929/ethz-c-000790876. **Canale:** DB (arXiv).
- **Modello:** VAE con encoder e decoder causali e prior dato da un flow (RealNVP). La loss è un upper bound della causal Wasserstein distance.
- **Dati:** Black-Scholes, Heston e volatilità path-dependent simulati; S&P 500 e VIX. **Task:** generazione di serie robuste per hedging, optimal stopping e opzioni americane.
- **Metriche:** CAUS (metriche adattate alla filtrazione); valori di problemi di controllo. **SF:** SF3.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 P, Q7 ?.

### Limmer & Horvath (2024) — Robust Hedging GANs
- **Bib:** Applied Mathematical Finance 31(3):164–201, DOI 10.1080/1350486X.2024.2440661. **Canale:** DB.
- **Modello:** gioco a tre tra generatore, discriminatore e hedger; regolarizzazione con signature-MMD e una palla di Wasserstein.
- **Task:** hedging robusto all'incertezza di modello. **Baseline:** deep hedging su modelli standard. **Metriche:** DEC (ES del P&L di copertura, fuori campione); PATH (sig-MMD).
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 N, Q7 ?.

### Ning, Jaimungal, Zhang & Bergeron (2023) — superfici IV senza arbitraggio con VAE
- **Bib:** **SIAM Journal on Financial Mathematics 14(4):1004–1027**. **⚠ AGGIORNAMENTO:** nel tracker la sede non è indicata. **Canale:** DB.
- **Modello:** VAE e CVAE (condizionato al VIX) sullo spazio dei parametri di SDE a regime switching e di processi additivi di Lévy. L'assenza di arbitraggio è garantita per costruzione; sulla calibrazione si usa un regolarizzatore di Wasserstein.
- **Dati:** opzioni FX. **Task:** scenari e rischio di coda per portafogli di opzioni. **Metriche:** distanza tra superfici fuori campione.
- **Q:** Q1 P, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 N, Q7 ?.

### Park, Ko & Lee (2025) — diffusion su grafici di prezzo
- **Bib:** Computational Economics 66:349–375, DOI 10.1007/s10614-024-10668-4. **Canale:** DB.
- **Modello:** DDPM con U-Net su immagini OHLC, poi riconvertite in serie numeriche.
- **Dati:** processi simulati (GBM, CEV, Heston, Merton); S&P 500; opzioni call giornaliere. **Task:** pricing Monte Carlo di opzioni.
- **Metriche:** test KS; errori di pricing.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 ?, Q5 Y, Q6 N, Q7 ?.

### Tan, Zhang, Zhao & Wang (2022) — DeepPricing
- **Bib:** Financial Innovation 8:64, DOI 10.1186/s40854-022-00369-y. **Canale:** DB.
- **Modello:** FinGAN, con TCN, che genera traiettorie risk-neutral usate in un Least-Squares Monte Carlo.
- **Dati:** obbligazioni convertibili cinesi. **Baseline:** Black-Scholes, CEV, GARCH, GAN con MLP e con LSTM.
- **Metriche:** sei fatti stilizzati; MAPE di pricing; DEC (strategie long-short e long-only). **SF:** SF1, SF2, SF3, SF5, SF6, SF8.
- **Q:** Q1 P, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 Y, Q7 ?.

### Vuletić & Cont (2024/2025) — VolGAN
- **Bib:** Applied Mathematical Finance 31(4):203–238, DOI 10.1080/1350486X.2025.2471317. **Canale:** DB.
- **Modello:** CGAN con penalità di regolarità (norma di Sobolev) e riponderazione degli scenari in base all'arbitraggio.
- **Dati:** volatilità implicite dell'SPX, 2000–2023. **Task:** scenari e hedging (selezione degli strumenti con LASSO).
- **Metriche:** autocorrelazione e mean reversion delle volatilità implicite; analisi a componenti principali; violazioni di arbitraggio; DEC (hedging). **SF:** SF3, SF11.
- **Q:** Q1 P, Q2 ?, Q3 Y (delta e delta-vega), Q4 N, Q5 Y, Q6 Y, Q7 ?.

### Bergeron, Fung, Hull & Poulos (2021/2022) — VAE per la volatilità (recuperato da Zotero)
- **Bib:** arXiv 2102.03945. Va citata la versione su JFDS (2022), da verificare. **Canale:** DB (arXiv).
- **Modello:** VAE con decoder a griglia e decoder "pointwise".
- **Dati:** superfici di volatilità implicita FX. **Baseline:** Heston. **Metriche:** MAE di ricostruzione e completamento.
- **Q:** Q1 P, Q2 ?, Q3 Y, Q4 N, Q5 Y, Q6 N, Q7 ?.

---

## Cartella 13 — Portafoglio, allocazione e covarianza (corpus)

### Cetingoz & Lehalle (2026) — "Synthetic data for portfolios: a throw of the dice will never abolish chance"
- **Bib:** **Quantitative Finance 26(8):1311–1342 (2026)**, DOI 10.1080/14697688.2026.2673155. **⚠ AGGIORNAMENTO:** nel tracker era indicato come preprint arXiv del 2025. **Canale:** DB (arXiv).
- **Modello:** pipeline a fattori (PCA e Random Matrix Theory) con GAN per i fattori e misture di t di Student per i residui.
- **Teoria:** generare più dati sintetici non riduce l'errore di stima ereditato dal campione di training.
- **Metodo di valutazione:** "regurgitative training" come test di identificabilità.
- **Dati:** 433 titoli dell'S&P 500. **Metriche:** DEC (Sharpe di strategie mean-reversion su più scale temporali). **SF:** SF1, SF9.
- **Q:** Q1 Y, Q2 ?, Q3 Y (bootstrap e parametrici), Q4 Y, Q5 Y, Q6 P, Q7 ?.

### Gopal (2024) — NeuralFactors
- **Bib:** **ACM ICAIF '24**, DOI 10.1145/3677052.3698647. **⚠ CORREZIONE:** l'anno è il 2024, non il 2023. **Canale:** DB.
- **Modello:** VAE con decoder strutturato a fattori (R = βF + ε).
- **Dati:** costituenti point-in-time dell'S&P 500. **Baseline:** PPCA, un generatore deep di base (BDG), GARCH.
- **Metriche:** PROB (log-verosimiglianza); RISK (calibrazione del VaR); DEC (portafogli). **SF:** SF1, SF9.
- **Q:** Q1 P, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 P, Q7 ?.
- **⚠ Affermazione verificata (p.2):** nel VaR il modello batte PPCA e BDG, ma **"GARCH still outperforms NeuralFactors in terms of calibration error"**. È un'altra evidenza per la Sez. 6.3.

### Lesniewski & Trigila (2024/2025) — Beyond Monte Carlo
- **Bib:** arXiv 2412.00036v3 (febbraio 2025). **Canale:** DB (SSRN).
- **Modello:** score-based diffusion, con lo score stimato tramite le equazioni di Kolmogorov e la quadratura di Gauss–Hermite.
- **Task:** scenari e stima della covarianza (numero di condizionamento). **Metriche:** condizionamento della matrice; DEC.
- **Q:** Q1 Y, Q2 ?, Q3 Y (covarianza campionaria), Q4 N, Q5 ?, Q6 N, Q7 ?.

### Lu & Yi (2022) — Autoencoding Conditional GAN
- **Bib:** arXiv 2207.05701. **Canale:** DB (SSRN).
- **Modello:** CGAN con un autoencoder nel generatore, addestrata come WGAN-GP.
- **Dati:** titoli USA ed EU. **Task:** allocazione mean-variance su dati sintetici.
- **Baseline:** Markowitz storico, CGAN. **Metriche:** DEC (Sharpe, rendimento cumulato); fatti stilizzati. **SF:** SF1, SF5.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 P, Q7 ?.

### Lu & Ding (2022) — HybridCGAN e HybridACGAN
- **Bib:** arXiv (2022). **Canale:** DB (SSRN).
- **Modello:** CGAN o ACGAN più un regressore ("proposer") che fornisce il trend.
- **Dati e metriche:** come Lu & Yi (2022).
- **Q:** come Lu & Yi (2022).

### Mariani, Zhu, Li, Scheidegger, Istrate, Bekas & Malossi (2019) — PAGAN
- **Bib:** arXiv 1909.10578 (IBM Research). **Canale:** DB (SSRN).
- **Modello:** DCGAN condizionale.
- **Dati:** universi azionari USA ed EU. **Task:** scenari per l'ottimizzazione mean-variance.
- **Baseline:** dati storici, bootstrap. **Metriche:** DEC (rendimenti mensili, volatilità, Sharpe); ripetibilità; mode collapse.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 N, Q5 Y, Q6 N, Q7 Y (ripetibilità).

### Marti (2020) — CorrGAN
- **Bib:** sede da verificare (probabilmente IEEE ICASSP 2020). **Canale:** DB (arXiv).
- **Modello:** DCGAN su matrici di correlazione riordinate con clustering gerarchico, poi proiettate sulla matrice di correlazione valida più vicina.
- **Dati:** rendimenti giornalieri di titoli dell'S&P 500 su un anno (252 giorni).
- **Metriche:** fatti stilizzati delle correlazioni (spettro degli autovalori, Perron–Frobenius, gerarchia, MST). **SF:** SF9.
- **Codice:** Y (secondo il log di ricerca).
- **Q:** Q1 Y, Q2 Y, Q3 P, Q4 N, Q5 NA, Q6 Y, Q7 N.

### Sun, Wu & Yan (2024) — GF-AGRU con ottimizzazione CVaR
- **Bib:** Journal of Economic Dynamics and Control 160 (2024) 104821. **Canale:** DB.
- **Modello:** modello fattoriale generativo con Attention-GRU e trasformazione per le code pesanti asimmetriche.
- **Dati:** componenti del Dow Jones. **Baseline:** equal weight, SAA statica, DCC-GARCH.
- **Metriche:** DEC e RISK (MDD, ES, IR, rapporto STARR, rapporto di Rachev). **SF:** SF1, SF6.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 N, Q5 Y, Q6 P, Q7 ?.

### Zhou, Yu, Gong & Huang (2025) — augmentation dei fattori con diffusion
- **Bib:** preprint (il tracker indica Finance Research Letters; da verificare). **Canale:** DB (SSRN).
- **Modello:** DDIM con backbone Transformer.
- **Dati:** fattori dall'Open Source Asset Pricing (Chen & Zimmermann). **Task:** augmentation per la stima della covarianza nel mean-variance.
- **Baseline:** DCC-NLS, statistica di Gerber, shrinkage.
- **Metriche:** DEC (rendimento, ricchezza finale, Sharpe, turnover) con test di significatività. **SF:** SF9.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 N, Q5 Y, Q6 N, Q7 Y.

### Noguer i Alonso (2025) — Normalizing Flows per il portafoglio (recuperato da Zotero)
- **Bib:** working paper AIFI (12/2/2025). **Canale:** DB (SSRN).
- **Modello:** normalizing flow della distribuzione congiunta dei rendimenti, usato per massimizzare lo Sharpe.
- **Dati:** dataset sintetico di 5 asset. **Metriche:** DEC (Sharpe).
- **Q:** Q1 Y, Q2 ?, Q3 P, Q4 N, Q5 N, Q6 N, Q7 N.
- **Nota:** evidenza debole (solo dati sintetici).

### Barro, Corazza & Filograsso (2025) [nuovo, canale DB]
- **Bib:** Working Paper Ca' Foscari, Dipartimento di Economia, n. 21/WP/2025.
- **Metodo:** scenari generati con WGAN-GP dentro un modello di doppio tracking error con mandato green.
- **Metriche:** DEC (Sharpe, Omega; intensità carbonica; fuori campione su 12 finestre).
- **Q:** Q1 P, Q2 ?, Q3 Y, Q4 N, Q5 Y, Q6 P, Q7 N.

### Kubiak, Weyde, Galkin, Philps & Gopal (2024) [nuovo, canale AM]
- **Bib:** ACM ICAIF '24, DOI 10.1145/3677052.3698640.
- **Modello:** DDPM incondizionato e condizionato per matrici di correlazione.
- **Baseline:** block bootstrap, WGAN. **Metriche:** fatti stilizzati delle correlazioni; regimi; caso di studio sull'allocazione. **SF:** SF9.
- **Codice:** Y (GitHub).
- **Q:** Q1 Y, Q2 Y, Q3 Y, Q4 Y, Q5 ?, Q6 Y, Q7 ?.

### Kim, Kim, Lee, Kim & Fabozzi (2025) [nuovo, canale AM]
- **Bib:** Annals of Operations Research 346:217–244, DOI 10.1007/s10479-024-06293-x.
- **Metodo:** anomaly score ottenuto da GAN, integrato nell'ottimizzazione mean-variance (shrinkage, statistica di Gerber).
- **Dati:** S&P 500 e 11 settori, 2000–2023. **Metriche:** DEC fuori campione.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 N, Q5 Y, Q6 N, Q7 ?.

---

## Cartella 14 — Generazione decision-aware e robusta (corpus; contiene la categoria di confine dei "generatori di decisioni")

### Kim, Kang, Jeon & Lee (2019) — GAN-MP
- **Bib:** *The Engineering Economist*, DOI 10.1080/0013791X.2019.1620391 (verificato su Scopus e sul PDF). **Canale:** DB.
- **⚠ CORREZIONE IMPORTANTE:** il tracker attribuisce GAN-MP a "Baek, Lee, Chun & Oh 2024, Expert Systems with Applications" e riporta dataset come Hang Seng, DAX, FTSE, S&P 100 e Nikkei. Il PDF e il record di Scopus sono invece **Kim, Kang, Jeon & Lee (2019)**. Dati e risultati vanno estratti di nuovo da questo PDF; vedi le righe sotto.
- **Metodo:** la GAN genera soluzioni candidate (pesi di portafoglio) che inizializzano un risolutore di programmazione matematica, per un mean-variance con vincoli reali (cardinalità e altri). È nella categoria di confine dei generatori di decisioni.
- **Dati:** titoli dell'S&P 500 (p.20).
- **Q:** Q1 Y, Q2 ?, Q3 Y (euristiche e risolutori), Q4 N, Q5 ?, Q6 NA, Q7 ?.

### Krach, Teichmann & Wutte (2025) — robust utility optimization con GAN
- **Bib:** arXiv 2403.15243v2 (settembre 2025). **Canale:** DB (arXiv).
- **Metodo:** gioco a somma zero tra la rete dell'investitore e quella del mercato, con perturbazioni dentro una palla di Wasserstein. I mercati di riferimento includono GARCH con rumore. Categoria di confine (generatore di scenari avversi).
- **Metriche:** DEC (utilità; confronto con soluzioni analitiche).
- **Codice:** Y (GitHub, p.13).
- **Q:** Q1 Y, Q2 Y, Q3 Y, Q4 N, Q5 Y, Q6 NA, Q7 ?.

### Montalván Hernández & de Campos (2023) [R] — Credal Probabilistic Circuits
- **Bib:** ACM ICAIF '23, DOI 10.1145/3604237.3626853. **Canale:** DB.
- **Modello:** circuiti probabilistici credali, cioè modelli generativi trattabili con insiemi di probabilità (probabilità imprecise).
- **Task:** portafoglio robusto. **Baseline:** Markowitz, equal weight, circuiti probabilistici standard.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 P, Q5 Y, Q6 NA, Q7 ?.
- **Nota:** è vicino ai temi di ricerca dei co-autori (probabilità imprecise); utile per la Sez. 7.3.

### Pun, Wang & Wong (2020) — GANr, "Financial Thought Experiment"
- **Bib:** SSRN (sede da verificare). **Canale:** DB (SSRN).
- **Metodo:** CLSGAN più una terza rete "regressore" che ottimizza i pesi; il parametro μ regola quanto avversi sono gli scenari.
- **Dati:** un anno di dati giornalieri, con bootstrap. **Baseline:** LASSO, CLSGAN. **Metriche:** DEC (quantili delle perdite, varianza).
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 NA, Q7 ?.

### Scholl, Mahfouz, Calinescu & Farmer (2025) — gestione di portafoglio oltre le funzioni di utilità
- **Bib:** ACM ICAIF '25. **⚠ CORREZIONE:** gli autori sono quattro (mancano Mahfouz e Farmer nel tracker). **Canale:** DB (arXiv).
- **Metodo:** GAN che impara la distribuzione condizionata dei pesi di portafoglio dei fondi, con un latente che rappresenta lo "stile" del gestore. Categoria di confine (behavioural cloning).
- **Q:** Q1 P, Q2 ?, Q3 Y, Q4 ?, Q5 Y, Q6 NA, Q7 ?.

### Sun, Deng, Chen & Parkes (2023) — DAT-CGAN
- **Bib:** ACM ICAIF '23, DOI 10.1145/3604237.3626855. **Canale:** DB.
- **Metodo:** loss multi-Wasserstein che include le grandezze rilevanti per la decisione; vengono forniti bound di generalizzazione.
- **Dati:** ETF. **Task:** scelta di portafoglio multi-periodo.
- **Baseline:** TimeGAN, RCGAN, bootstrap. **Metriche:** DEC (qualità delle decisioni prese sui dati sintetici).
- **Q:** Q1 Y, Q2 ?, Q3 Y (bootstrap), Q4 Y, Q5 Y, Q6 N, Q7 ?.

### Visentin & Cheridito (2025) — G-causal Normalizing Flows
- **Bib:** arXiv 2510.15458 (17/10/2025). **Canale:** DB (arXiv).
- **Teoria e metodo:** i problemi di ottimizzazione robusta agli interventi sono continui rispetto alla distanza di Wasserstein G-causale, mentre possono non esserlo rispetto a quella standard. I flow G-causali minimizzano la distanza G-causale.
- **Task:** augmentation per un mean-variance in un modello a fattori causale. **Metriche:** CAUS; DEC.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 Y, Q5 Y, Q6 NA, Q7 ?.

### Zelman, Stefanik, Weiss & Teichmann (2024) [R] — Adversarial IRL per il market making
- **Bib:** ACM ICAIF '24, DOI 10.1145/3677052.3698641. **Canale:** DB.
- **Metodo:** AIRL, in cui il generatore produce azioni di market making. Categoria di confine dei generatori di decisioni.
- **Metriche:** backtest fuori campione. **Codice:** Y (GitHub).
- **Q:** Q1 N, Q2 Y, Q3 Y, Q4 N, Q5 Y, Q6 NA, Q7 ?.

### Ang, Bao, Huang, Wang, Xi, Lu, Tung & Huang (2025) — CMS-VAE [nuovo, canale AM]
- **Bib:** ACM ICAIF '25.
- **Modello:** VAE con encoder convoluzionale dilatato e una supervisione "strategy-aware": loss sulla distanza tra le metriche di rischio e rendimento reali e sintetiche (VaR, ES, MDD, PnL).
- **Dati:** cripto multi-asset. **Baseline:** modelli generativi stato dell'arte (dalla linea di lavoro di CTBench).
- **Metriche:** fedeltà; utilità predittiva (errori fino al 96,8% più bassi); DEC (arbitraggio statistico, Sharpe 1,4 volte superiore). **SF:** SF1, SF3, SF9.
- **Q:** Q1 Y, Q2 ?, Q3 ?, Q4 Y, Q5 Y, Q6 P, Q7 ?.

---

## Cartella 15 — Backtesting e ambienti di RL (corpus)

### Ang, Wang, Huang, Bao, Xi, Tung, Jin & Huang (2026) — CTBench
- **Bib:** **ICLR 2026**. **⚠ AGGIORNAMENTO:** non è anonimo e non è più "under review". **Canale:** AM (incluso con l'opzione B).
- **Benchmark:** 452 token cripto, rendimenti orari; due task: utilità predittiva (IC, IR, VaR, ES) e arbitraggio statistico (momentum e top-quantile weighting).
- **Modelli confrontati:** Quant-GAN, COSCI-GAN, TimeVAE, KoVAE, Diffusion-TS, FIDE, Fourier-Flow, LS4, più le **baseline classiche ARMA-GARCH e Bootstrap**.
- **Protocollo:** training su 500 giorni, test su 30 o 15 giorni, con riaddestramento.
- **Codice:** Y (github.com/MilleXi/CTBench).
- **Affermazioni verificate (p.6):**
  - "In 2021 (Bull Market), **Diffusion-TS achieves the best forecasting error yet fails** [to generate] tradable signals";
  - il ranking cambia con il regime di mercato;
  - "Classical baselines ARMA-GARCH and Bootstrap maintain **mid-tier** performance".
- **Q:** Q1 Y, Q2 Y, Q3 Y, Q4 Y, Q5 Y, Q6 P, Q7 Y.

### Aghapour, Bayraktar & Yuan (2025) — selezione dinamica di portafoglio con diffusion
- **Bib:** NeurIPS 2025 Workshop "Generative AI in Finance". **Canale:** DB (SSRN).
- **Metodo:** diffusion score-based (VP-SDE) con encoder RNN e campionamento adattivo; l'agente TD3 è addestrato sull'ambiente generativo. Bound di stabilità nella distanza di Wasserstein adattata AW2.
- **Dati:** 10 Industry Portfolios. **Baseline:** Markowitz, equal weight, S&P 500. **Metriche:** DEC (Sharpe); CAUS (AW2).
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 N, Q5 Y, Q6 N, Q7 ?.

### Kazadaev, Pozdnyakov & Makarov (2024) — GAN e momentum sulla Borsa di Mosca
- **Bib:** IEEE CIFEr 2024. **Canale:** DB.
- **Modello:** GAN con TCN su 5 titoli del MOEX in 10 anni. **Baseline:** bootstrap.
- **Task:** valutare e ottimizzare una strategia momentum. **Metriche:** correlazioni, distribuzioni; DEC (momentum).
- **Affermazione verificata (p.2):** le GAN riproducono le correlazioni semplici, ma "may fail to capture weak dependencies" come il momentum.
- **Q:** Q1 Y, Q2 ?, Q3 Y, Q4 N, Q5 Y, Q6 P, Q7 ?.

### Koshiyama, Firoozye & Treleaven (2021) — GAN per il fine-tuning e la combinazione di strategie
- **Bib:** Quantitative Finance 21(5):797–813, DOI 10.1080/14697688.2020.1790635. **Canale:** DB.
- **Modello:** cGAN con MLP.
- **Dati:** 579 asset (azioni, indici, valute, swap). **Baseline:** stationary bootstrap, k-fold, block CV.
- **Metriche:** DEC (Sharpe, Calmar). **SF:** SF2.
- **Q:** Q1 P, Q2 N, Q3 Y, Q4 N, Q5 Y, Q6 P, Q7 Y.

### Lezmi, Roche, Roncalli & Xu (2020) — robustezza del backtesting con macchine di Boltzmann e GAN
- **Bib:** arXiv 2007.04838 (Amundi). Gli autori oltre ai primi due sono da verificare. **Canale:** DB (SSRN).
- **Modelli:** RBM gaussiane e condizionali; WGAN; CDCWGAN.
- **Dati:** S&P 500 e VIX; futures azionari e obbligazionari. **Task:** backtest di una strategia risk parity e distribuzione del max drawdown.
- **Baseline:** bootstrap. **Metriche:** momenti, copula, ACF. **SF:** SF1, SF2, SF9.
- **Q:** Q1 Y, Q2 N, Q3 Y, Q4 Y, Q5 NA, Q6 Y, Q7 N.

### Liu (2024) — RL, GAN e Transformer per portafoglio e HFT
- **Bib:** atti della Greater Bay Area International Conference on Digital Economy and AI (ACM). **Canale:** DB.
- **Metriche:** rendimento, Sharpe, VaR, CVaR. Il reporting è debole: il ruolo della GAN è descritto in modo vago.
- **Q:** Q1 N, Q2 N, Q3 P, Q4 N, Q5 ?, Q6 N, Q7 N.

### Liu, Ventre & Polukarov (2022) — augmentation sintetica per il deep RL nel trading [nuovo, canale DB]
- **Bib:** ACM ICAIF '22, DOI 10.1145/3533271.3561704.
- **Metodo:** TimeGAN genera titoli Google e Bitcoin giornalieri; agenti PPO e A2C addestrati con FinRL.
- **Metriche:** TSTR; DEC (rendimento annuo, Sharpe, Sortino).
- **Q:** Q1 Y, Q2 ?, Q3 N, Q4 P, Q5 Y, Q6 N, Q7 ?.

---

## Nuovi studi inclusi dopo la ri-valutazione (cartelle 01, 02, 05)

### Takahashi, Chen & Tanaka-Ishii (2019) — FIN-GAN [canale AM, cartella 01]
- **Bib:** Physica A 527 (2019) 121261.
- **Modello:** GAN con generatori MLP e MLP-CNN.
- **Dati:** 486 titoli dell'S&P 500 con almeno 1.000 giorni di storico, più indici. **Task:** dati sintetici.
- **Metriche:** sei fatti stilizzati (imprevedibilità lineare, code pesanti, volatility clustering, leverage effect, correlazione coarse-fine, asimmetria gain/loss). **SF:** SF1, SF2, SF3, SF5, SF6, SF8.
- **Codice:** Y (github.com/stakahashy/fingan).
- **Q:** Q1 Y, Q2 Y, Q3 N (gli ARCH/GARCH sono discussi ma non usati come baseline), Q4 P (confronto tra architetture), Q5 NA, Q6 Y, Q7 P.
- **Nota:** è lo studio che ha fissato la "batteria" di fatti stilizzati poi ripresa da Tan et al. 2022 e da altri.

### Huang, Chen & Qiao (2024) — FTS-Diffusion [canale AM, cartella 01]
- **Bib:** ICLR 2024.
- **Modello:** tre moduli: riconoscimento di pattern invarianti di scala (clustering di sequenze con DTW), diffusion per i segmenti, evoluzione dei pattern tramite una catena di Markov.
- **Dati:** S&P 500 e altri. **Task:** dati sintetici e augmentation; l'errore di previsione si riduce fino al 17,9%.
- **Baseline:** TimeGAN, RCGAN e altre. **Metriche:** fatti stilizzati; TSTR. **SF:** SF1, SF3.
- **Codice:** Y (materiale supplementare).
- **Q:** Q1 Y, Q2 Y, Q3 ?, Q4 Y, Q5 Y, Q6 Y, Q7 ?.

### Wang & Hirsa (2024) — Advanced GANs for financial time series [canale DB, cartella 01]
- **Bib:** preprint SSRN (29/7/2024), 137 pagine (Columbia University).
- **Modelli:** maschera di attention stocastica applicata a TAGAN e TTGAN; HLF-GAN; RegimeGAN (HMM più GAN); CGAN; VQC-VAEGAN.
- **Dati:** benchmark e futures. **Metriche:** fatti stilizzati (distribuzioni, ACF). **SF:** SF1, SF2, SF3, SF11.
- **Q:** Q1 Y, Q2 ?, Q3 ?, Q4 Y, Q5 NA, Q6 Y, Q7 ?.
- **Nota:** lo stesso gruppo ha prodotto Fu et al. 2022 e MSSTD-Diff.

### Hosseini, Niccolai & Grimaccia (2025) — quote di emissione UE ad alta frequenza [canale DB, cartella 02]
- **Bib:** IEEE CIFEr Companion 2025.
- **Modello:** WGAN-GP con LSTM e self-attention, applicata a sequenze di pattern.
- **Dati:** quote di emissione EU ETS, tick ricampionati a 5 minuti; dati confidenziali. **Metriche:** fatti stilizzati (curtosi, clustering, leverage, imprevedibilità). **SF:** SF1, SF2, SF3, SF5.
- **Codice:** Y (GitHub; i dati non sono condivisi).
- **Q:** Q1 N, Q2 Y, Q3 N, Q4 N, Q5 NA, Q6 Y, Q7 N.

### Abdelhady (Salama), Dahmani, AbouEl-Magd, Darwish & Hassanien (2024) — crescita della finanza green [canale DB, cartella 05; caso borderline]
- **Bib:** PLOS ONE 19(7):e0306874.
- **Modello:** augmentation con CT-GAN e previsione con una rete NARNN.
- **Dati:** crescita della finanza green per paese; fuori campione 2004–2009. **Metriche:** MAPE.
- **Q:** Q1 P, Q2 N, Q3 P, Q4 N, Q5 Y, Q6 N, Q7 N.

> **Correzione aggiunta (verifica successiva):** Lim & Gorse (2021) usa l'**order flow BTC-USD di Coinbase** (p.4), non 5 titoli tecnologici come indica il tracker.
