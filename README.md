# Proyecto de Clasificación de Audio: SVM y Regresión Logística

Este proyecto implementa modelos de **Máquina de Vectores de Soporte (SVM)** y **Regresión Logística** para clasificar datos proporcionalmente desbalanceados con clases predominantes de audio de toses de personas con covid como "positivos" y "negativos" a covid-19. Se incluye la carga y procesamiento de los datos, entrenamiento de los modelos, evaluación de rendimiento y validación cruzada. 

## Contenido

- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Descripción de los Modelos](#descripción-de-los-modelos)
- [Evaluación del Rendimiento](#evaluación-del-rendimiento)
- [Validación Cruzada y Bootstrap](#validación-cruzada-y-bootstrap)
- [Referencias](#referencias)

## Instalación

```bash
pip install numpy pandas matplotlib seaborn librosa cvxopt pywavelets
```
## Estructura del proyecto
1. **Preprocesamiento**:
   - Balanceo de clases mediante reducción de datos negativos y técnicas como SMOTE.
   - Transformaciones de características utilizando la transformada de `Fourier` y `MFCC`.

2. **Entrenamiento de Modelos**:
   - **SVM**: Implementación desde cero, incluyendo normalización, funciones de pérdida y derivadas, y actualización de parámetros.
   - **Regresión Logística**: Implementación de funciones de costo, derivadas, y ajuste de parámetros.

3. **Evaluación de Modelos**:
   - Matriz de confusión, precisión, recall, y F1-score.
   - Validación cruzada K-Fold y Bootstrap.

## Descripción de los Modelos

### Máquina de Vectores de Soporte (SVM)

El modelo de SVM está implementado desde cero, incluyendo:

- **Normalización**: Usando `RobustScaler`.
- **Función de Pérdida**: La función de pérdida se calcula como una combinación de regularización y margen de error.
- **Derivadas y Actualización de Parámetros**: Ajustes en los parámetros `w` y `b` mediante gradiente descendente.

### Regresión Logística

La regresión logística se implementa con `regularización L2` y ajuste de hiperparámetros usando `BorderlineSMOTE` para el balanceo de datos. La función sigmoide se usa para la predicción, y se aplica un umbral de 0.15 para clasificar los datos.

## Evaluación del Rendimiento

- **Matriz de Confusión**: Generada para visualizar el rendimiento del modelo SVM y de regresión logística.
- **Métricas de Clasificación**: Incluye precisión, recall, F1-score y exactitud.

## Validación Cruzada y Bootstrap

- **Validación Cruzada K-Fold**: Para obtener métricas más confiables. Se utiliza un rango de valores `K`, graficando la precisión para observar el impacto de K.
- **Bootstrap**: Estimación de precisión mediante remuestreo con la librería `scipy`.