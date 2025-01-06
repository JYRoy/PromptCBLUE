#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

import sys
sys.path.append("../../")
sys.path.append("./")

from src.ft_chatglm_ptuning.tokenization_chatglm import ChatGLMTokenizer
from src.ft_chatglm_ptuning.configuration_chatglm import ChatGLMConfig
from src.ft_chatglm_ptuning.modeling_chatglm import ChatGLMForConditionalGeneration
from src.ft_chatglm_lora.trainer_seq2seq import Seq2SeqTrainer
from src.ft_chatglm_lora.arguments import ModelArguments, DataTrainingArguments

from peft import PeftModel, LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict

logger = logging.getLogger(__name__)

def main():
    # 创建一个 HfArgumentParser 对象，用于解析模型、数据训练和序列到序列训练的参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    # 如果命令行参数的长度为 2 且第二个参数以.json 结尾
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # 如果我们只向脚本传递一个参数，并且它是一个 json 文件的路径，解析该文件以获取我们的参数
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # 否则，从命令行参数解析为数据类
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    # 配置日志记录
    logging.basicConfig(
        # 日志的格式，包括时间、日志级别、日志名称和消息内容
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        # 时间的格式
        datefmt="%m/%d/%Y %H:%M:%S",
        # 日志处理器，将日志输出到标准输出流
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # 如果训练参数中指示应该记录日志
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        # 默认的训练日志级别是 passive，将其设置为 info 级别
        transformers.utils.logging.set_verbosity_info()

    # 获取进程的日志级别
    log_level = training_args.get_process_log_level()
    # 设置日志记录器的日志级别
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    # 启用默认的日志处理器
    transformers.utils.logging.enable_default_handler()
    # 启用明确的日志格式
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    # 在每个进程中记录一个简短的摘要
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # 记录训练/评估参数
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    # 在初始化模型之前设置随机种子
    set_seed(training_args.seed)

    # Load dataset
    # 用于存储数据文件的字典，根据不同的数据类型存储不同的文件路径
    data_files = {}
    # 如果训练数据文件不为空
    if data_args.train_file is not None:
        # 将训练文件添加到 data_files 字典中，并将键设置为 "train"
        data_files["train"] = data_args.train_file
        # 获取训练文件的扩展名
        extension = data_args.train_file.split(".")[-1]
    # 如果验证数据文件不为空
    if data_args.validation_file is not None:
        # 将验证文件添加到 data_files 字典中，并将键设置为 "validation"
        data_files["validation"] = data_args.validation_file
        # 获取验证文件的扩展名
        extension = data_args.validation_file.split(".")[-1]
    # 如果测试数据文件不为空
    if data_args.test_file is not None:
        # 将测试文件添加到 data_files 字典中，并将键设置为 "test"
        data_files["test"] = data_args.test_file
        # 获取测试文件的扩展名
        extension = data_args.test_file.split(".")[-1]
    # 从指定的 JSON 文件加载数据集
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print("raw_datasets: ", raw_datasets)
    # print("raw_datasets: ", len(raw_datasets["train"]))

    # Load pretrained model and tokenizer
    config = ChatGLMConfig.from_pretrained(
        model_args.model_name_or_path,
        # trust_remote_code=True
    )
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    tokenizer = ChatGLMTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # trust_remote_code=True
    )
    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    ).half().cuda()  # 转换为半精度并移动到 GPU 上

    # 如果有预训练的 peft 路径
    if model_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        # 从预训练的模型和 peft 路径加载 Peft 模型
        model = PeftModel.from_pretrained(model, model_args.peft_path)
    else:
        logger.info("Init new peft model")
        # 将可训练的模块列表按照逗号分割
        target_modules = model_args.trainable.split(',')
        # 将需要保存的模块列表按照逗号分割，如果不为 "null" 则分割，否则为 None
        modules_to_save = model_args.modules_to_save.split(',') if model_args.modules_to_save!="null" else None
        # 创建 LoRA 配置
        lora_rank = model_args.lora_rank
        lora_dropout = model_args.lora_dropout
        lora_alpha = model_args.lora_alpha
        print(target_modules)
        print(lora_rank)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save
        )
        # 基于 LoRA 配置对模型进行适配
        model = get_peft_model(model, peft_config)
    # 打印模型的可训练参数
    model.print_trainable_parameters()

    # 获取源前缀，如果没有则为空
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    def preprocess_function_eval(examples):
        import pdb; pdb.set_trace()
        # 存储输入和目标的列表
        inputs, targets = [], []
        # 遍历示例中的 prompt 列
        for i in range(len(examples[prompt_column])):
            # 如果 response 列为空，则将目标添加为 "filled in!"
            if not examples[response_column][i]:
                targets.append("filled in !")
            else:
                # 否则添加实际的 response 作为目标
                targets.append(examples[response_column][i])
            # 如果 prompt 列不为空
            if examples[prompt_column][i]:
                query = examples[prompt_column][i]
                # 如果没有历史列或历史列为空
                if history_column is None or len(examples[history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    # 构建历史信息的提示
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                # 将构建好的提示添加到输入列表
                inputs.append(prompt)
        # 在输入前添加前缀
        inputs = [prefix + inp for inp in inputs]
        # 使用分词器对输入进行处理，设置最大长度，进行截断和填充
        model_inputs = tokenizer(inputs,
                                 max_length=data_args.max_source_length,
                                 truncation=True,
                                 padding=True)
        # 使用分词器对目标进行处理，设置最大长度，进行截断
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
        # 如果忽略填充标记的损失
        if data_args.ignore_pad_token_for_loss:
            # 将填充标记的 id 替换为 -100
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        # 将处理好的标签添加到模型输入中
        model_inputs["labels"] = labels["input_ids"]
        # 返回处理好的模型输入
        return model_inputs

    def preprocess_function_train(examples):
        # 将 examples[prompt_column] 和 examples[response_column] 作为 query 和 answer 转换为 id
        # examples 的数据类型：<class 'datasets.formatting.formatting.LazyBatch'>
        import pdb; pdb.set_trace()
        # 计算最大序列长度，为源和目标的最大长度之和
        max_seq_length = data_args.max_source_length + data_args.max_target_length
        # 存储输入 id 和标签的字典
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        # 遍历示例中的 prompt 列
        for i in range(len(examples[prompt_column])):
            # 如果 prompt 列和 response 列都不为空
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]
                # 如果没有历史列
                if history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    # 构建历史信息的提示
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                # 在提示前添加前缀
                prompt = prefix + prompt
                # 对提示进行编码，不添加特殊标记
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                # 对答案进行编码，不添加特殊标记
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)
                # 如果提示长度超过最大源长度，截断
                if len(a_ids) > data_args.max_source_length - 1:
                    a_ids = a_ids[: data_args.max_source_length - 1]
                # 如果答案长度超过最大目标长度，截断
                if len(b_ids) > data_args.max_target_length - 2:
                    b_ids = b_ids[: data_args.max_target_length - 2]
                # 构建包含特殊标记的输入 id，[CLS]和[SEP]
                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
                # 找到上下文的长度，即推理开始标记的位置，130004
                context_length = input_ids.index(tokenizer.bos_token_id)
                # 找到掩码的位置
                mask_position = context_length - 1
                # 构建标签，上下文部分为 -100，其余部分为输入 id
                labels = [-100] * context_length + input_ids[mask_position+1:]
                # 计算填充长度
                pad_len = max_seq_length - len(input_ids)
                # 对输入 id 进行填充
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                # 对标签进行填充
                labels = labels + [tokenizer.pad_token_id] * pad_len
                # print("input_ids: ", len(input_ids))
                # print("labels: ", len(labels))
                # 如果忽略填充标记的损失
                if data_args.ignore_pad_token_for_loss:
                    # 将填充标记的 id 替换为 -100，这里对应到 compute_metrics 中 labels 去除 pad token
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
                # 将处理好的输入 id 添加到输入列表
                model_inputs["input_ids"].append(input_ids)
                # 将处理好的标签添加到标签列表
                model_inputs["labels"].append(labels)
        # 返回处理好的模型输入
        return model_inputs
    
    def print_dataset_example(example):
        print("input_ids: ",example["input_ids"])
        print("inputs: ", tokenizer.decode(example["input_ids"]))
        print("label_ids: ", example["labels"])
        print("labels: ", tokenizer.decode(example["labels"]))

    if training_args.do_train:
        # 检查是否存在训练数据集，如果不存在则抛出异常
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        # 获取训练数据集
        train_dataset = raw_datasets["train"]
        # 如果设置了最大训练样本数，则根据实际数据集长度和最大样本数取较小值作为样本数
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # 确保在主进程中首先执行数据集的映射预处理操作，将文本转换为 token ids
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            # 使用 map 函数对训练数据集进行预处理，调用 preprocess_function_train 函数
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        # 打印第一个训练数据集示例的信息
        print_dataset_example(train_dataset[0])
        # 打印第二个训练数据集示例的信息
        print_dataset_example(train_dataset[1])

    if training_args.do_eval:
        # 为评估设置最大目标长度
        max_target_length = data_args.val_max_target_length
        # 检查是否存在验证数据集，如果不存在则抛出异常
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        # 获取验证数据集
        eval_dataset = raw_datasets["validation"]
        # 如果设置了最大评估样本数，则根据实际数据集长度和最大样本数取较小值作为样本数
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        # 确保在主进程中首先执行数据集的映射预处理操作
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            # 使用 map 函数对验证数据集进行预处理，调用 preprocess_function_eval 函数
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
            )
        # 打印第一个验证数据集示例的信息
        print_dataset_example(eval_dataset[0])
        # 打印第二个验证数据集示例的信息
        print_dataset_example(eval_dataset[1])

    if training_args.do_predict:
        # 为预测设置最大目标长度
        max_target_length = data_args.val_max_target_length
        # 检查是否存在测试数据集，如果不存在则抛出异常
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        # 获取测试数据集
        predict_dataset = raw_datasets["test"]
        # 如果设置了最大预测样本数，则根据实际数据集长度和最大样本数取较小值作为样本数
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        # 确保在主进程中首先执行数据集的映射预处理操作
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            # 使用 map 函数对预测数据集进行预处理，调用 preprocess_function_eval 函数
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
            )
        # 打印第一个预测数据集示例的信息
        print_dataset_example(predict_dataset[0])
        # 打印第二个预测数据集示例的信息
        print_dataset_example(predict_dataset[1])

    # Data collator
    # 根据是否忽略填充标记的损失，设置标签填充标记的 id
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # 创建数据整理器对象，用于整理序列到序列任务的数据
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

    # Metric
    def compute_metrics(eval_preds):
        # 获取预测结果和标签
        preds, labels = eval_preds
        # 如果预测结果是元组，取第一个元素
        if isinstance(preds, tuple):
            preds = preds[0]
        # 对预测结果进行批量解码，跳过特殊标记
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
         # 如果忽略填充标记的损失，将标签中 -100 替换为填充标记的 id
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # 对标签进行批量解码，跳过特殊标记
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # 存储不同评估指标的分数
        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        # 遍历解码后的预测结果和标签
        for pred, label in zip(decoded_preds, decoded_labels):
            # 使用 jieba 分词对预测结果和标签进行分词
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            # 创建 Rouge 评估器
            rouge = Rouge()
            hypothesis = ' '.join(hypothesis)
            # 如果预测结果为空，设置为 "-"
            if not hypothesis:
                hypothesis = "-"
            # 计算 Rouge 分数
            scores = rouge.get_scores(hypothesis, ' '.join(reference))
            result = scores[0]
            # 将 Rouge 分数添加到分数字典中
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            # 计算 BLEU 分数
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))
        # 计算 BLEU 分数
        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        # 返回分数字典
        return score_dict

    # Override the decoding parameters of Seq2SeqTrainer
    # 如果没有设置生成的最大长度，使用验证集的最大目标长度
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    # 如果没有设置生成的束搜索数量，使用数据参数中的束搜索数量
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    # Initialize our Trainer
    # 初始化序列到序列训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        save_prefixencoder=model_args.pre_seq_len is not None
    )

    # Training
    if training_args.do_train:
        # 检查是否有检查点
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        # 启用梯度检查点
        model.gradient_checkpointing_enable()
        # 启用输入梯度要求
        model.enable_input_require_grads()
        # 获取训练指标
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload
        # 获取训练指标
        metrics = train_result.metrics
        # 计算使用的训练样本数
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        # 记录和保存训练指标
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # 进行评估，设置评估指标前缀，启用采样，设置采样参数
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=512, temperature=0.95)
        # 计算使用的评估样本数
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        # 记录和保存评估指标
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # 读取原test file
        list_test_samples = []
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                list_test_samples.append(line)
        # 进行预测，设置预测指标前缀，设置预测参数
        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            # max_tokens=512,
            max_new_tokens=data_args.max_target_length,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            # top_p=0.7,
            # temperature=0.95,
            # repetition_penalty=1.1
        )
        # 获取预测指标
        metrics = predict_results.metrics
        print(metrics)
        # 计算使用的预测样本数
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        # 记录和保存预测指标
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                # 对预测结果进行解码
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                # 对标签进行解码
                labels = tokenizer.batch_decode(
                    predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip() for label in labels]
                assert len(labels) == len(list_test_samples)
                # 存储预测结果的文件路径
                output_prediction_file = os.path.join(training_args.output_dir, "test_predictions.json")
                # 将预测结果写入文件
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for idx, (p, l) in enumerate(zip(predictions, labels)):
                        samp = list_test_samples[idx]
                        samp["target"] = p
                        res = json.dumps(samp, ensure_ascii=False)
                        writer.write(f"{res}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
