from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import warnings


# Подавляем все предупреждения
warnings.filterwarnings("ignore")

# Настройка конфигурации BitsAndBytes для квантования в 4 бита
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Загрузка модели с использованием новой конфигурации
model = AutoModelForCausalLM.from_pretrained(
    'openchat/openchat-3.5-0106',
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map='cuda:0'
)


tokenizer = AutoTokenizer.from_pretrained('openchat/openchat-3.5-0106')
pipe_openchat = pipeline('text-generation',
                         model=model,
                         tokenizer=tokenizer)
