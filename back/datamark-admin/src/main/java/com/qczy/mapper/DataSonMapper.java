package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.model.request.DataSonQueryRequest;
import com.qczy.model.response.DataMarkResponse;
import com.qczy.model.response.DataSonResponse;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 21:45
 * @Description:
 */
public interface DataSonMapper extends BaseMapper<DataSonEntity> {

    List<DataSonResponse> selectDataSonByFatherId(@Param("groupId") String groupId);


    /**
     * 批量删除【请填写功能名称】
     *
     * @param list 需要删除的【请填写功能名称】主键集合
     * @return 结果
     */
    public int deleteDataSonByIds(@Param("list") List<Integer> list);


    IPage<DataMarkResponse> getDataSetMarkList(Page<DataMarkResponse> pageParam, @Param("req") DataSonQueryRequest request);


    /**
     *  根据sonId 获取 数据集
     */
    @Select("select * from qczy_data_son where son_id = #{sonId}")
    DataSonEntity  getDataSonBySonId(String sonId);


    Long countCurrentNodeFiles(Long id);
}
