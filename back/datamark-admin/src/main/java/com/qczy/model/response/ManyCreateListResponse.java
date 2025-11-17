package com.qczy.model.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/27 10:16
 * @Description:
 */
@Data
public class ManyCreateListResponse {

    /**
     * 任务id
     */
    private Integer id;

    /**
     * 任务名称
     */
    private String taskName;

    /**
     * 数据集id
     */
    private String sonId;

    /**
     * 来源数据集
     */
    private String sonName;

    /**
     * 标注类型
     */
    private String anoType;

    /**
     * 任务状态
     */
    private String taskState;

    /**
     * 创建时间
     */
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date createTime;

    /**
     *  团队人数
     */
    private Integer userNum;

    /**
     *  文件数量
     */
    private Integer fileNum;

    /**
     *  是否允许删除  0:不允许 1：可以删除
     */
    private Integer endStatus;

    /**
     *  团队名称
     */
    private String teamName;

}
