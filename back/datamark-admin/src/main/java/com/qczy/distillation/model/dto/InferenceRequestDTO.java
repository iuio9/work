package com.qczy.distillation.model.dto;

import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotNull;

/**
 * 模型推理请求DTO
 *
 * @author AI Assistant
 * @date 2025-01-13
 */
@Data
@ApiModel("模型推理请求")
public class InferenceRequestDTO {

    @ApiModelProperty(value = "训练任务ID（使用训练好的模型）", required = true)
    @NotBlank(message = "训练任务ID不能为空")
    private String taskId;

    @ApiModelProperty(value = "输入图像目录", required = true)
    @NotBlank(message = "输入图像目录不能为空")
    private String inputDir;

    @ApiModelProperty(value = "输出JSON目录", required = true)
    @NotBlank(message = "输出JSON目录不能为空")
    private String outputDir;

    @ApiModelProperty(value = "批次大小", example = "8")
    private Integer batchSize = 8;

    @ApiModelProperty(value = "数据集ID（可选，用于将结果保存到数据集）")
    private String datasetId;

    @ApiModelProperty(value = "是否自动导入标注结果", example = "true")
    private Boolean autoImport = true;
}
