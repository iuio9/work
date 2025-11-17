package com.qczy.controller.dataset;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.mapper.DataSonMapper;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.model.request.*;
import com.qczy.model.response.DataDetailsResponse;
import com.qczy.model.response.DataResponse;
import com.qczy.service.DataFatherService;
import com.qczy.service.DataSonDetailsService;
import com.qczy.service.DataSonService;
import com.qczy.service.ManyMarkService;
import com.qczy.utils.StringUtils;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.ObjectUtils;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;


/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 21:46
 * @Description: 数据集管理
 */
@RestController
@RequestMapping("/data/set")
@Api(tags = "数据集管理")
public class DataSetController {


    @Autowired
    private DataSonService dataSonService;

    @Autowired
    private DataFatherService dataFatherService;

    @Autowired
    private DataSonDetailsService dataSonDetailsService;

    @Autowired
    private ManyMarkService manyMarkService;


    @GetMapping("/getDataSetList")
    @ApiOperation("数据集列表")
    public Result getDataSetList(@RequestParam Integer page,
                                 @RequestParam Integer limit,
                                 @ModelAttribute DataSonQueryRequest request) {
        Page<DataResponse> pageParam = new Page<>(page, limit);
        IPage<DataResponse> dataSetList = dataSonDetailsService.getDataSetList(pageParam, request);
        return Result.ok(dataSetList);
    }


    /**
     * 数据集列表不分页
     */
    @GetMapping("/getDataSetListNoPage")
    @ApiOperation("数据集列表不分页")
    public Result getDataSetListNoPage() {
        return Result.ok(dataSonDetailsService.getDataSetListNoPage());
    }


    /**
     * 新增保存【请填写功能名称】
     */
    @PostMapping("/add")
    @ApiOperation("新增数据集")
    public Result addSave(@RequestBody DataSonEntityRequest dataSonRequest) {
        Map<String, Object> map = new HashMap<>();
        map.put("status", dataSonService.insertDataSet(dataSonRequest));
        if (!StringUtils.isEmpty(dataSonRequest.getFileIds())) {
            map.put("groupId", dataSonRequest.getFatherId());
            map.put("sonId", dataSonRequest.getSonId());
        }
        return Result.ok(map);
    }


    /**
     * 新增数据集版本
     */
    @PostMapping("/addDataVersion")
    @ApiOperation("新增数据集版本")
    public Result addDataVersion(@RequestBody SaveSonVersionRequest saveSonVersionRequest) {
        Map<String, Object> map = new HashMap<>();
        map.put("status", dataSonService.addDataVersion(saveSonVersionRequest));
        if (saveSonVersionRequest.getIsInherit() == 1) {
            map.put("groupId", saveSonVersionRequest.getGroupId());
            DataSonEntity entity = dataSonService.getOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getFatherId, saveSonVersionRequest.getGroupId())
                            .eq(DataSonEntity::getVersion, saveSonVersionRequest.getNewVersion())
            );
            map.put("sonId", entity.getSonId());
        }
        return Result.ok(map);
    }

    /**
     * 数据集导入
     */
    @PostMapping("/dataSetImport")
    @ApiOperation("导入")
    public Result dataSetImport(@RequestBody DataSonEntityRequest dataSonRequest) {
        if (ObjectUtils.isEmpty(dataSonRequest)) {
            return Result.fail("数据集信息不能为空！");
        }
        if (StringUtils.isEmpty(dataSonRequest.getSonId()) && StringUtils.isEmpty(dataSonRequest.getFileIds())) {
            return Result.fail("传入的数据集对象不能为空！");
        }
        Map<String, Object> map = new HashMap<>();
        map.put("status", dataFatherService.dataSetImport(dataSonRequest));
        if (!StringUtils.isEmpty(dataSonRequest.getFileIds())) {
            DataSonEntity entity = dataSonService.getOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, dataSonRequest.getSonId())
            );
            map.put("groupId", entity.getFatherId());
            map.put("sonId", dataSonRequest.getSonId());
        }
        return Result.ok(map);
    }


    @DeleteMapping("/deleteDataGroup")
    @ApiOperation("删除数据集组")
    public Result deleteDataGroup(@RequestParam String groupId) throws IOException {
        if (StringUtils.isBlank(groupId)) {
            return Result.fail("数据集组id不能为空！");
        }
        // 判断当前数据集租下的数据级是否在执行任务
        if (manyMarkService.countBySonIdsTask(groupId) > 0) {
            return Result.fail("当前数据集组下的数据集正在执行任务，请先删除任务后重试！");
        }
        return Result.ok(dataFatherService.deleteDataGroup(groupId));
    }


    @DeleteMapping("/deleteDataSet")
    @ApiOperation("删除单个数据集")
    public Result deleteDataSet(@RequestParam String sonId) throws IOException {
        if (StringUtils.isBlank(sonId)) {
            return Result.fail("数据集id不能为空！");
        }
        if (manyMarkService.countBySonIdTask(sonId) > 0) {
            return Result.fail("当前数据集正在执行任务，请先删除任务后重试！");
        }
        return Result.ok(dataFatherService.deleteDataSet(sonId));
    }


    @PostMapping("/updateDataSetName")
    @ApiOperation("修改数据集组名称")
    public Result updateDataSetName(@RequestBody UpdateDataSetRequest request) {
        if (ObjectUtils.isEmpty(request)) {
            return Result.fail("传入的参数对象不能为空！");
        }
        if (StringUtils.isEmpty(request.getGroupId()) || StringUtils.isEmpty(request.getGroupName())) {
            return Result.fail("传入的数据集组id 或者 数据集名称 不能为空！");
        }
        return Result.ok(dataFatherService.updateDataSetName(request));
    }


    @PostMapping("/updateDataSetRemark")
    @ApiOperation("修改数据集组名称")
    public Result updateDataSetRemark(@RequestBody UpdateDataSetRequest request) {
        if (ObjectUtils.isEmpty(request)) {
            return Result.fail("传入的参数对象不能为空！");
        }
        if (StringUtils.isEmpty(request.getSonId()) || StringUtils.isEmpty(request.getRemark())) {
            return Result.fail("传入的数据集组id 或者 数据集名称 不能为空！");
        }
        return Result.ok(dataSonService.updateDataSetRemark(request));
    }


    @GetMapping("/getDataDetails")
    @ApiOperation("查看")
    public Result getDataDetails(
            @RequestParam Integer page,
            @RequestParam Integer limit,
            @RequestParam String sonId,
            @RequestParam Integer state,
            Integer labelId,
            Integer markUserId,
            Integer taskId,
            String sign) {
        if (StringUtils.isEmpty(sonId)) {
            return Result.fail("数据集ID不能为空！");
        }
        Page<DataDetailsResponse> pageParam = new Page<>(page, limit);
        IPage<DataDetailsResponse> detailsRequestList = dataSonDetailsService.getDataDetails(pageParam, sonId, state, labelId, markUserId, taskId, sign);
        return Result.ok(detailsRequestList);
    }


    /**
     * 查看 原始图
     *
     * @param sonId
     * @return
     */
    @GetMapping("/getDataDetailsNoMarkFilePath")
    @ApiOperation("查看原始图")
    public Result getDataDetailsNoMarkFilePath(
            @RequestParam Integer page,
            @RequestParam Integer limit,
            @RequestParam String sonId,
            @RequestParam Integer state,
            Integer markUserId,
            Integer taskId,
            String sign) {
        if (StringUtils.isEmpty(sonId)) {
            return Result.fail("数据集ID不能为空！");
        }
        Page<DataDetailsResponse> pageParam = new Page<>(page, limit);
        IPage<DataDetailsResponse> detailsRequestList = dataSonDetailsService.getDataDetailsNoMarkFilePath(pageParam, sonId, state, markUserId, taskId, sign);
        return Result.ok(detailsRequestList);
    }


    @GetMapping("/DataDetailsCount")
    @ApiOperation("查看（全部、有标注信息、五标注信息）分页总数量")
    public Result DataDetailsCount(@RequestParam String sonId, Integer markUserId, Integer taskId, String sign) {
        if (StringUtils.isEmpty(sonId)) {
            return Result.fail("数据集ID不能为空！");
        }
        Map<String, Integer> data = new HashMap<>();
        data.put("all", dataSonDetailsService.selectFileAndlabelCount(sonId, markUserId, taskId, sign));
        data.put("haveAno", dataSonDetailsService.selectFileAndlabelYesMarkCount(sonId, markUserId, taskId, sign));
        data.put("noAno", dataSonDetailsService.selectFileAndlabelNoMarkCount(sonId, markUserId, taskId, sign));
        data.put("invalid", dataSonDetailsService.selectFileInvalidDataCount(sonId, markUserId, taskId, sign));
        return Result.ok(data);
    }

    @PostMapping("/getResultDataSetSave")
    @ApiOperation("根据结果创建数据集")
    public Result getResultDataSetSave(@RequestBody ResultDataSonRequest request, BindingResult bindingResult) {
        if (ObjectUtils.isEmpty(request)) {
            return Result.fail("请求对象不能为空!");
        }
        if (request.getTaskId() == null && StringUtils.isEmpty(request.getGroupName()) && request.getDataTypeId() == null) {
            return Result.fail("参数不能为空！");
        }
        return Result.ok(dataSonService.getResultDataSetSave(request));
    }

    // 获取当前数据集的类型
    @GetMapping("/getDataSonType")
    public Result getDataSonType(@RequestParam String sonId) {
        if (StringUtils.isEmpty(sonId)) {
            return Result.fail("数据集id不能为空！");
        }
        DataSonEntity entity = dataSonService.getOne(
                new QueryWrapper<DataSonEntity>().eq("son_id", sonId)
        );
        if (entity == null) {
            return Result.fail("数据集不存在！");
        }
        return Result.ok(entity);
    }


    @Autowired
    private DataSonMapper dataSonMapper;


    @GetMapping("test")
    public Result test() {
        return Result.ok(dataSonMapper.selectList(null));
    }


}
