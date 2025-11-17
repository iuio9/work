package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.qczy.model.entity.ManyFileEntity;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/12 9:51
 * @Description:
 */
public interface ManyFileMapper extends BaseMapper<ManyFileEntity> {

    // 查询否和条件的文件
    List<ManyFileEntity> selectBatchByTaskAndFileIds( @Param("taskId") Integer taskId,
                                                      @Param("fileIds") List<Integer> fileIds,
                                                      @Param("userId") Integer userId);

    // 批量新增
    void insertBatch(List<ManyFileEntity> list);

    // 批量修改
    void updateBatchByIds(@Param("entities") List<ManyFileEntity> entities);

}
