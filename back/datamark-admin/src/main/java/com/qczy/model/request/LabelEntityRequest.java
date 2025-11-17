package com.qczy.model.request;

import com.qczy.model.entity.LabelEntity;
import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/27 11:19
 * @Description:
 */
@Data
public class LabelEntityRequest extends LabelEntity {

    private String sonId;

    private String twoLabelName;

}
