# SynkData Bot — Agente de Ventas Inteligente

Bot de ventas para SynkData Technologies con RAG, memoria de sesión y personalidad humana.

## Stack
- **Backend**: FastAPI + Anthropic Claude Haiku + ChromaDB (RAG) + pypdf
- **Frontend**: HTML/CSS/JS standalone con estética SynkData
- **Deploy**: Docker + docker-compose (Railway / Azure / VPS)

---

## Setup local rápido

### 1. Variables de entorno
Crea un archivo `.env` en la raíz:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
ADMIN_PASSWORD=tu_contraseña_segura
```

### 2. Levantar con Docker Compose
```bash
docker-compose up --build
```

El bot queda en: http://localhost:8000

---

## Deploy en Railway

1. Conecta el repo a Railway
2. Configura las variables de entorno:
   - `ANTHROPIC_API_KEY`
   - `ADMIN_PASSWORD`
3. Railway detecta el `Dockerfile` en `/backend` automáticamente
4. Agrega un volumen persistente en `/app/data` para que el ChromaDB no se borre

---

## Deploy en Azure (App Service)

```bash
# Build y push a Azure Container Registry
az acr build --registry <tu-registry> --image synkdata-bot .

# Deploy con volumen persistente
az webapp create --resource-group <rg> --plan <plan> \
  --name synkdata-bot --deployment-container-image-name <imagen>
```

---

## Panel Admin (RAG)

1. Abre el bot y haz clic en "⚙️ Panel Admin (RAG)"
2. Ingresa tu `ADMIN_PASSWORD`
3. Sube documentos `.txt`, `.md` o `.pdf` con información de tus servicios
4. El bot automáticamente usa esa info en cada conversación

### Documentos recomendados para cargar:
- `ventaspro.md` — descripción completa de VentasPro
- `imss-intel.md` — qué hace el Laboratorio IMSS-Intel
- `precios.md` — rangos de precios y tiempos de entrega
- `casos-de-exito.md` — proyectos anteriores y resultados
- `faq.md` — preguntas frecuentes de clientes

---

## Endpoints API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/chat` | Conversación con el bot |
| POST | `/api/contact` | Guardar datos de lead |
| POST | `/api/admin/upload` | Subir doc al RAG |
| POST | `/api/admin/documents` | Listar docs indexados |
| DELETE | `/api/admin/documents` | Eliminar doc del RAG |
| GET | `/api/admin/contacts` | Ver leads capturados |

---

## Personalizar el System Prompt

Edita `backend/main.py` → `BASE_SYSTEM_PROMPT` para:
- Cambiar el nombre del consultor (actualmente "Carlos Reyes")
- Ajustar los servicios descritos
- Cambiar el tono o las expresiones usadas
- Agregar reglas de negocio específicas

---

## Modelo de IA

Por defecto usa `claude-haiku-4-5` (el más económico de Anthropic).
Para más calidad, cambia a `claude-sonnet-4-6` en `main.py` línea del `messages.create()`.

Costo estimado con Haiku: ~$0.002 USD por conversación promedio.
