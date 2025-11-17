package com.qczy.model.request;

import lombok.Data;
import org.jpedal.parser.shape.S;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/16 15:21
 * @Description:
 */
@Data
public class ModelApplyForRequestParam {

    /**
     * 模型名称
     */
    private String modelName;


    /**
     * 模型来源
     */
    private String modelSource;


    /**
     * 建设单位名称
     */
    private String buildUnitName;


    /**
     * 承建单位名称
     */
    private String btUnitName;

    /**
     * 申请时间 - 开始
     */
    private Date applyForDateStart;


    /**
     * 申请时间 - 结束
     */
    private Date applyForDateEnd;

    /**
     * 昵称
     */
    private String nickName;
}
