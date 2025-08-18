# Curso Procesamiento del Lenguaje Natural 2 – Trabajos Prácticos

Este repositorio contiene tres trabajos prácticos desarrollados para la materia de **Procesamiento de Lenguaje Natural 2**. Cada TP aborda un desafío distinto, desde la modificación de un modelo base hasta la construcción de un sistema multiagente.

---

## Trabajo Práctico 1 – TinyGPT + Mixture of Experts (MoE)
En este trabajo se parte de **TinyGPT**, un modelo liviano basado en GPT, y se lo modifica para que soporte la arquitectura **Mixture of Experts (MoE)**.  
El objetivo es mejorar la generación de texto mediante la incorporación de múltiples expertos, controlados por un **gating network** que decide qué expertos participan en cada paso de inferencia.

**Puntos clave:**
- Implementación de la capa de *experts*.
- Gating network con proyección lineal.
- Evaluación del modelo modificado vs. el TinyGPT original.

[Notebook de TP1](tp_1.ipynb)

---

## Trabajo Práctico 2 – Chatbot con RAG + Pinecone
En este trabajo se construye un **chatbot basado en Retrieval-Augmented Generation (RAG)**.  
El chatbot utiliza **Pinecone** como base vectorial para indexar y recuperar información del **CV del alumno**.  
De esta forma, una **LLM** puede responder preguntas de manera más contextualizada y precisa sobre la información contenida en el CV.

**Puntos clave:**
- Indexación de documentos en Pinecone.
- Construcción del pipeline RAG.
- Respuestas de la LLM basadas en la información indexada.

[Notebook de TP2](tp_2.ipynb)

---

## Trabajo Práctico 3 – Chatbot Multiagente con múltiples CVs
A partir del TP2, se extiende el chatbot para que soporte **múltiples CVs**.  
Cada CV es manejado por un **agente independiente**, de manera que el sistema funciona como un **multiagente**, donde cada agente responde preguntas específicas sobre la persona de su CV.

**Puntos clave:**
- Carga dinámica de múltiples CVs.
- Creación de agentes especializados por persona.
- Coordinación entre agentes para responder preguntas.

[Notebook de TP3](tp_3.ipynb)

---

## Ejecución del Chatbot

El chatbot puede correrse con diferentes **flags** que controlan el comportamiento del sistema.

### Ejemplo de uso:
```
python chatbot/app.py --create-index True --reloader False --upload-data True
```

### Flags disponibles:

`--create-index` (bool, default=False)

Si está en True, crea los índices de Pinecone desde cero.

`--reloader` (bool, default=False)

Controla si se utiliza el reloader de Flask. Útil en desarrollo para evitar ejecuciones duplicadas.

`--upload-data` (bool, default=False)

Si está en True, sube los datos (CVs) a la base vectorial para su indexación.

---

## Videos

Se grabaron videos dónde se muestran los distintos casos de uso de los trabajos 2 y 3. 

[Carpeta con archivos de video](chatbot/screen_captures)

---

## Requisitos:

- Python 3.9+
- Flask
- Pinecone
- Hugging Face Transformers
- Torch

---

## Autor

[Ing. Christian Pisani Testa](mailto:christiantpg@gmail.com)

Nu. SIU: A1715
