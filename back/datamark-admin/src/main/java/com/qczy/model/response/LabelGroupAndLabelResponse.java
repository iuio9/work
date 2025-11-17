package com.qczy.model.response;

import com.qczy.model.entity.LabelEntity;
import lombok.Data;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/9/29 17:27
 * @Description:
 */
@Data
public class LabelGroupAndLabelResponse {
    private Integer id;
    private String labelGroupName;
    private List<LabelEntity> list;
}
