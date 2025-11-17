package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.FileEntity;
import com.qczy.model.request.GetMarkInfoRequest;
import com.qczy.model.response.DataDetailsResponse;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 22:41
 * @Description:
 */
public interface FileMapper extends BaseMapper<FileEntity> {
    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteFileByIds(String[] ids);

    // --------- 带分页信息-------------
    // 全部
    IPage<DataDetailsResponse> selectFileAndlabel(Page<DataDetailsResponse> pageParam, @Param("fileIds") String fileIds, @Param("labelId") Integer labelId);

    // 有标注信息
    IPage<DataDetailsResponse> selectFileAndlabelYesMark(Page<DataDetailsResponse> pageParam, @Param("fileIds") String fileIds, @Param("labelId") Integer labelId);

    // 无标注信息
    IPage<DataDetailsResponse> selectFileAndlabelNoMark(Page<DataDetailsResponse> pageParam, @Param("fileIds") String fileIds, @Param("labelId") Integer labelId);

    // 无效数据
    IPage<DataDetailsResponse> selectFileInvalidData(Page<DataDetailsResponse> pageParam, @Param("fileIds") String fileIds, @Param("labelId") Integer labelId);

    // --------- 不带分页信息-------------
    List<DataDetailsResponse> selectFileAndlabelNoPage(@Param("fileIds") String fileIds);

    List<DataDetailsResponse> selectFileAndlabelYesMarkNoPage(@Param("fileIds") String fileIds);

    List<DataDetailsResponse> selectFileAndlabelNoMarkNoPage(@Param("fileIds") String fileIds);

    int selectFileAndlabelCount(@Param("fileIds") String fileIds);

    int selectFileAndlabelYesMarkCount(@Param("fileIds") String fileIds);

    int selectFileAndlabelNoMarkCount(@Param("fileIds") String fileIds);

    int selectFileInvalidDataCount(@Param("fileIds") String fileIds);


    // --------------- 查看当前图片是否标注 -----------------
    Integer getFileNoMark(@Param("fileId") Integer fileId);

    // --------------- 查看手动标注和厂商标注的数据列表 -----------------
    IPage<DataDetailsResponse> selectFileMarkInfoAndlabel(Page<DataDetailsResponse> pageParam,
                                                          @Param("fileIds") String fileIds,
                                                          @Param("labelId") Integer labelId,
                                                          @Param("taskId") Integer taskId);


    List<DataDetailsResponse> selectFileAndlabelVersionTwo(
            @Param("accessAddress") String accessAddress,
            @Param("groupId") String groupId,
            @Param("versionFormat") String versionFormat,
            @Param("version") Integer version,
            @Param("separator") String separator,
            @Param("fileIds") String fileIds);


    // 根据sonId 查询数据集文件列表
    List<FileEntity> getFileListBySonId(String sonId);

}
