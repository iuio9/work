package com.qczy.distillation.model.dto;

import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

/**
 * 模型推理结果DTO
 *
 * @author AI Assistant
 * @date 2025-01-13
 */
@Data
@ApiModel("模型推理结果")
public class InferenceResultDTO {

    @ApiModelProperty("推理任务ID")
    private String inferenceId;

    @ApiModelProperty("训练任务ID")
    private String taskId;

    @ApiModelProperty("模型类型")
    private String modelType;

    @ApiModelProperty("处理图像数量")
    private Integer processedImages;

    @ApiModelProperty("成功数量")
    private Integer successCount;

    @ApiModelProperty("失败数量")
    private Integer failureCount;

    @ApiModelProperty("输出目录")
    private String outputDir;

    @ApiModelProperty("推理状态")
    private String status;

    @ApiModelProperty("错误信息")
    private String errorMessage;

    @ApiModelProperty("开始时间")
    private String startTime;

    @ApiModelProperty("结束时间")
    private String endTime;

    @ApiModelProperty("耗时（秒）")
    private Long duration;
}
