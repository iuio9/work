package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.qczy.model.entity.DataSonLabelEntity;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/22 14:52
 * @Description:
 */
public interface DataSonLabelMapper extends BaseMapper<DataSonLabelEntity> {

    // 查询当前数据集有没有绑定当前的这个标签---> labelName
    int selectBySonIdAndLabelNameCount(@Param("sonId") String sonId, @Param("labelName") String labelName);

    void insertBatch(List<DataSonLabelEntity> insertList);

}
