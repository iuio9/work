package com.qczy.model.request;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.util.List;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-27 15:20
 * @description：
 * @modified By：
 * @version: $
 */
@Data
public class SegmentParams {
    @ApiModelProperty(value = "背景点")
    private int[][] background_points;
    @ApiModelProperty(value = "前景点")
    private int[][] foreground_points;
    @ApiModelProperty(value = "文件路径")
    private String image_path;
}
