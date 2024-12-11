use skia::gpu::{self, gl::FramebufferInfo, DirectContext};
use skia::Contains;
use skia_safe as skia;
use std::collections::HashMap;
use uuid::Uuid;

use crate::debug;
use crate::math::Rect;
use crate::shapes::{draw_image_in_container, Fill, Image, Kind, Shape};
use crate::view::Viewbox;

static ROBOTO_REGULAR: &[u8] = include_bytes!("fonts/RobotoMono-Regular.ttf");
static TYPEFACE_ALIAS: &str = "roboto-regular";

struct GpuState {
    pub context: DirectContext,
    framebuffer_info: FramebufferInfo,
}

impl GpuState {
    fn new() -> Self {
        let interface = skia::gpu::gl::Interface::new_native().unwrap();
        let context = skia::gpu::direct_contexts::make_gl(interface, None).unwrap();
        let framebuffer_info = {
            let mut fboid: gl::types::GLint = 0;
            unsafe { gl::GetIntegerv(gl::FRAMEBUFFER_BINDING, &mut fboid) };

            FramebufferInfo {
                fboid: fboid.try_into().unwrap(),
                format: skia::gpu::gl::Format::RGBA8.into(),
                protected: skia::gpu::Protected::No,
            }
        };

        GpuState {
            context,
            framebuffer_info,
        }
    }

    /// Create a Skia surface that will be used for rendering.
    fn create_target_surface(&mut self, width: i32, height: i32) -> skia::Surface {
        let backend_render_target =
            gpu::backend_render_targets::make_gl((width, height), 1, 8, self.framebuffer_info);

        gpu::surfaces::wrap_backend_render_target(
            &mut self.context,
            &backend_render_target,
            skia::gpu::SurfaceOrigin::BottomLeft,
            skia::ColorType::RGBA8888,
            None,
            None,
        )
        .unwrap()
    }
}

pub(crate) struct CachedSurfaceImage {
    pub image: Image,
    pub viewbox: Viewbox,
    has_all_shapes: bool,
}

impl CachedSurfaceImage {
    fn is_dirty(&self, viewbox: &Viewbox) -> bool {
        !self.has_all_shapes && !self.viewbox.area.contains(viewbox.area)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct RenderOptions {
    debug_flags: u32,
    dpr: Option<f32>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            debug_flags: 0x00,
            dpr: None,
        }
    }
}

impl RenderOptions {
    pub fn is_debug_visible(&self) -> bool {
        self.debug_flags & debug::DEBUG_VISIBLE == debug::DEBUG_VISIBLE
    }

    pub fn dpr(&self) -> f32 {
        self.dpr.unwrap_or(1.0)
    }
}

pub(crate) struct RenderState {
    gpu_state: GpuState,
    pub final_surface: skia::Surface,
    pub drawing_surface: skia::Surface,
    pub debug_surface: skia::Surface,
    pub cached_surface_image: Option<CachedSurfaceImage>,
    options: RenderOptions,
    pub viewbox: Viewbox,
    images: HashMap<Uuid, Image>,
}

impl RenderState {
    pub fn new(width: i32, height: i32) -> RenderState {
        // This needs to be done once per WebGL context.
        let mut gpu_state = GpuState::new();
        let mut final_surface = gpu_state.create_target_surface(width, height);
        let drawing_surface = final_surface
            .new_surface_with_dimensions((width, height))
            .unwrap();
        let debug_surface = final_surface
            .new_surface_with_dimensions((width, height))
            .unwrap();

        RenderState {
            gpu_state,
            final_surface,
            drawing_surface,
            debug_surface,
            cached_surface_image: None,
            options: RenderOptions::default(),
            viewbox: Viewbox::new(width as f32, height as f32),
            images: HashMap::with_capacity(2048),
        }
    }

    pub fn add_image(&mut self, id: Uuid, image_data: &[u8]) -> Result<(), String> {
        let image_data = skia::Data::new_copy(image_data);
        let image = Image::from_encoded(image_data).ok_or("Error decoding image data")?;

        self.images.insert(id, image);
        Ok(())
    }

    pub fn has_image(&mut self, id: &Uuid) -> bool {
        self.images.contains_key(id)
    }

    pub fn set_debug_flags(&mut self, debug: u32) {
        self.options.debug_flags = debug;
    }

    pub fn set_dpr(&mut self, dpr: f32) {
        if Some(dpr) != self.options.dpr {
            self.options.dpr = Some(dpr);
            self.resize(
                self.viewbox.width.floor() as i32,
                self.viewbox.height.floor() as i32,
            );
        }
    }

    pub fn resize(&mut self, width: i32, height: i32) {
        let dpr_width = (width as f32 * self.options.dpr()).floor() as i32;
        let dpr_height = (height as f32 * self.options.dpr()).floor() as i32;

        let surface = self.gpu_state.create_target_surface(dpr_width, dpr_height);
        self.final_surface = surface;
        self.drawing_surface = self
            .final_surface
            .new_surface_with_dimensions((dpr_width, dpr_height))
            .unwrap();
        self.debug_surface = self
            .final_surface
            .new_surface_with_dimensions((dpr_width, dpr_height))
            .unwrap();

        self.viewbox.set_wh(width as f32, height as f32);
    }

    pub fn flush(&mut self) {
        self.gpu_state
            .context
            .flush_and_submit_surface(&mut self.final_surface, None)
    }

    pub fn translate(&mut self, dx: f32, dy: f32) {
        self.drawing_surface.canvas().translate((dx, dy));
    }

    pub fn scale(&mut self, sx: f32, sy: f32) {
        self.drawing_surface.canvas().scale((sx, sy));
    }

    pub fn reset_canvas(&mut self) {
        self.drawing_surface
            .canvas()
            .clear(skia::Color::TRANSPARENT)
            .reset_matrix();
        self.final_surface
            .canvas()
            .clear(skia::Color::TRANSPARENT)
            .reset_matrix();
        self.debug_surface
            .canvas()
            .clear(skia::Color::TRANSPARENT)
            .reset_matrix();
    }

    pub fn render_single_shape(&mut self, shape: &Shape) {
        // Check transform-matrix code from common/src/app/common/geom/shapes/transforms.cljc
        let mut matrix = skia::Matrix::new_identity();
        let (translate_x, translate_y) = shape.translation();
        let (scale_x, scale_y) = shape.scale();
        let (skew_x, skew_y) = shape.skew();

        matrix.set_all(
            scale_x,
            skew_x,
            translate_x,
            skew_y,
            scale_y,
            translate_y,
            0.,
            0.,
            1.,
        );

        let mut center = shape.selrect.center();
        matrix.post_translate(center);
        center.negate();
        matrix.pre_translate(center);

        self.drawing_surface.canvas().concat(&matrix);

        for fill in shape.fills().rev() {
            self.render_fill(fill, shape.selrect, &shape.kind);
        }

        let mut paint = skia::Paint::default();
        paint.set_blend_mode(shape.blend_mode.into());
        paint.set_alpha_f(shape.opacity);
        self.drawing_surface.draw(
            &mut self.final_surface.canvas(),
            (0.0, 0.0),
            skia::SamplingOptions::new(skia::FilterMode::Linear, skia::MipmapMode::Nearest),
            Some(&paint),
        );
        self.drawing_surface
            .canvas()
            .clear(skia::Color::TRANSPARENT);
    }

    pub fn navigate(&mut self, shapes: &HashMap<Uuid, Shape>) -> Result<(), String> {
        if let Some(cached_surface_image) = self.cached_surface_image.as_ref() {
            if cached_surface_image.is_dirty(&self.viewbox) {
                self.render_all(shapes, true);
            } else {
                self.render_all_from_cache()?;
            }
        }

        Ok(())
    }

    pub fn render_all(
        &mut self,
        shapes: &HashMap<Uuid, Shape>,
        generate_cached_surface_image: bool,
    ) {
        self.reset_canvas();
        self.scale(
            self.viewbox.zoom * self.options.dpr(),
            self.viewbox.zoom * self.options.dpr(),
        );
        self.translate(self.viewbox.pan_x, self.viewbox.pan_y);

        let is_complete = self.render_shape_tree(&Uuid::nil(), shapes);
        if generate_cached_surface_image || self.cached_surface_image.is_none() {
            self.cached_surface_image = Some(CachedSurfaceImage {
                image: self.final_surface.image_snapshot(),
                viewbox: self.viewbox,
                has_all_shapes: is_complete,
            });
        }

        if self.options.is_debug_visible() {
            self.render_debug();
        }

        self.flush();
    }

    fn render_fill(&mut self, fill: &Fill, selrect: Rect, kind: &Kind) {
        match (fill, kind) {
            (Fill::Image(image_fill), kind) => {
                let image = self.images.get(&image_fill.id());
                if let Some(image) = image {
                    draw_image_in_container(
                        &self.drawing_surface.canvas(),
                        &image,
                        image_fill.size(),
                        kind,
                        &fill.to_paint(&selrect),
                    );
                }
            }
            (_, Kind::Rect(rect)) => {

              // <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" height = "2560" width = "2560">
              //       <path d="M30,1h40l29,29v40l-29,29h-40l-29-29v-40z" stroke="#;000" fill="none"/>
              //       <path d="M31,3h38l28,28v38l-28,28h-38l-28-28v-38z" fill="#a23"/>
              //       <text x="50" y="68" font-size="48" fill="#FFF" text-anchor="middle"><![CDATA[410]]></text>
              //       <image x="100" y="100" width="256" height="256" xlink:href="data:image/gif;base64,R0lGODdhMAAwAPAAAAAAAP///ywAAAAAMAAwAAAC8IyPqcvt3wCcDkiLc7C0qwyGHhSWpjQu5yqmCYsapyuvUUlvONmOZtfzgFzByTB10QgxOR0TqBQejhRNzOfkVJ+5YiUqrXF5Y5lKh/DeuNcP5yLWGsEbtLiOSpa/TPg7JpJHxyendzWTBfX0cxOnKPjgBzi4diinWGdkF8kjdfnycQZXZeYGejmJlZeGl9i2icVqaNVailT6F5iJ90m6mvuTS4OK05M0vDk0Q4XUtwvKOzrcd3iq9uisF81M1OIcR7lEewwcLp7tuNNkM3uNna3F2JQFo97Vriy/Xl4/f1cf5VWzXyym7PHhhx4dbgYKAAA7"/>
              //       </svg>

                let svg = r##"<svg width="168.484" xmlns="http://www.w3.org/2000/svg" height="259" viewBox="1526.421 987.496 168.484 259" fill="none"><path d="M1607.704,987.496L1607.704,987.496ZZZZZZC1608.819,987.496,1609.931,987.523,1611.043,987.578C1612.154,987.633,1613.264,987.714,1614.371,987.823C1615.478,987.933,1616.583,988.069,1617.684,988.232C1618.785,988.395,1619.881,988.586,1620.972,988.803C1622.063,989.020,1623.149,989.263,1624.228,989.534C1625.308,989.804,1626.380,990.101,1627.445,990.424C1628.510,990.747,1629.566,991.096,1630.614,991.471C1631.662,991.846,1632.700,992.246,1633.728,992.672C1634.756,993.098,1635.773,993.549,1636.779,994.025C1637.785,994.500,1638.779,995.001,1639.760,995.525C1640.742,996.050,1641.710,996.598,1642.664,997.170C1643.619,997.743,1644.559,998.338,1645.484,998.956C1646.409,999.574,1647.319,1000.215,1648.213,1000.878C1649.107,1001.541,1649.984,1002.225,1650.844,1002.931C1651.704,1003.637,1652.546,1004.364,1653.371,1005.111C1654.196,1005.859,1655.002,1006.626,1655.789,1007.412C1656.576,1008.199,1657.343,1009.005,1658.090,1009.830C1658.837,1010.654,1659.564,1011.497,1660.270,1012.357C1660.976,1013.217,1661.660,1014.094,1662.323,1014.988C1662.986,1015.882,1663.627,1016.792,1664.245,1017.717C1664.863,1018.642,1665.458,1019.582,1666.031,1020.537C1666.603,1021.491,1667.151,1022.459,1667.676,1023.441C1668.201,1024.422,1668.701,1025.416,1669.176,1026.422C1669.652,1027.428,1670.103,1028.445,1670.529,1029.473C1670.955,1030.501,1671.355,1031.539,1671.730,1032.587C1672.105,1033.635,1672.454,1034.691,1672.777,1035.756C1673.100,1036.821,1673.397,1037.893,1673.667,1038.973C1673.938,1040.052,1674.181,1041.138,1674.398,1042.229C1674.615,1043.321,1674.806,1044.417,1674.969,1045.518C1675.132,1046.618,1675.269,1047.723,1675.378,1048.830C1675.487,1049.938,1675.568,1051.047,1675.623,1052.158C1675.678,1053.270,1675.705,1054.382,1675.705,1055.495L1675.705,1086.073C1676.298,1092.525,1677.280,1102.304,1678.161,1106.850C1679.538,1113.954,1683.855,1111.611,1683.855,1121.716C1683.855,1131.821,1679.527,1135.216,1679.508,1144.201C1679.499,1148.429,1682.907,1152.249,1686.519,1156.298C1690.583,1160.854,1694.905,1165.699,1694.905,1171.739C1694.905,1183.145,1687.831,1236.972,1617.283,1236.972C1601.056,1236.972,1588.283,1230.345,1578.232,1220.139C1579.757,1233.287,1580.705,1246.496,1580.705,1246.496L1533.705,1246.496C1534.179,1234.404,1532.286,1225.024,1530.312,1215.239C1528.404,1205.787,1526.421,1195.957,1526.421,1182.942C1526.421,1166.792,1531.489,1146.169,1535.366,1130.393C1537.848,1120.291,1539.843,1112.176,1539.705,1108.496C1539.725,1107.924,1539.745,1107.215,1539.766,1106.386C1539.725,1105.423,1539.705,1104.460,1539.705,1103.497L1539.705,1055.495C1539.705,1051.603,1540.037,1047.739,1540.700,1043.904C1540.702,1043.703,1540.704,1043.574,1540.705,1043.522C1540.705,1043.505,1540.705,1043.496,1540.705,1043.496L1540.706,1043.498C1540.711,1043.506,1540.727,1043.536,1540.756,1043.588C1540.929,1042.615,1541.123,1041.647,1541.338,1040.682C1541.553,1039.718,1541.789,1038.759,1542.046,1037.804C1542.304,1036.850,1542.581,1035.902,1542.880,1034.960C1543.178,1034.018,1543.497,1033.083,1543.836,1032.155C1544.175,1031.227,1544.534,1030.307,1544.914,1029.395C1545.293,1028.482,1545.692,1027.578,1546.111,1026.683C1546.529,1025.788,1546.967,1024.903,1547.424,1024.027C1547.882,1023.151,1548.358,1022.285,1548.853,1021.430C1549.348,1020.575,1549.861,1019.731,1550.393,1018.898C1550.925,1018.065,1551.475,1017.245,1552.042,1016.436C1552.610,1015.627,1553.195,1014.831,1553.797,1014.048C1554.399,1013.264,1555.018,1012.494,1555.654,1011.738C1556.290,1010.982,1556.942,1010.240,1557.610,1009.512C1558.278,1008.784,1558.962,1008.071,1559.661,1007.372C1560.361,1006.674,1561.075,1005.992,1561.804,1005.325C1562.533,1004.658,1563.276,1004.007,1564.034,1003.372C1564.791,1002.738,1565.562,1002.120,1566.346,1001.519C1567.131,1000.918,1567.928,1000.335,1568.737,999.768C1569.547,999.202,1570.369,998.654,1571.203,998.123C1572.036,997.593,1572.881,997.081,1573.737,996.587C1574.593,996.094,1575.459,995.619,1576.336,995.163C1577.213,994.707,1578.099,994.271,1578.995,993.854C1579.890,993.437,1580.795,993.039,1581.708,992.661C1582.621,992.284,1583.542,991.926,1584.471,991.588C1585.400,991.251,1586.335,990.933,1587.277,990.637C1588.220,990.340,1589.168,990.064,1590.123,989.808C1591.078,989.553,1592.037,989.318,1593.002,989.104C1593.967,988.891,1594.935,988.698,1595.908,988.527C1596.881,988.356,1597.858,988.205,1598.838,988.077C1599.817,987.948,1600.799,987.840,1601.784,987.754C1602.768,987.668,1603.754,987.604,1604.741,987.561C1605.728,987.518,1606.716,987.496,1607.704,987.496ZZZZZZZM1607.204,1019.496L1608.206,1019.496C1609.098,1019.496,1609.989,1019.518,1610.880,1019.562C1611.771,1019.605,1612.660,1019.671,1613.548,1019.758C1614.435,1019.846,1615.320,1019.955,1616.203,1020.086C1617.085,1020.217,1617.963,1020.369,1618.838,1020.543C1619.713,1020.717,1620.583,1020.913,1621.448,1021.129C1622.313,1021.346,1623.173,1021.584,1624.026,1021.843C1624.879,1022.102,1625.726,1022.381,1626.566,1022.682C1627.406,1022.982,1628.238,1023.303,1629.062,1023.645C1629.886,1023.986,1630.701,1024.347,1631.507,1024.729C1632.314,1025.110,1633.110,1025.511,1633.897,1025.931C1634.684,1026.352,1635.459,1026.791,1636.224,1027.250C1636.989,1027.708,1637.743,1028.185,1638.484,1028.681C1639.225,1029.176,1639.954,1029.690,1640.671,1030.221C1641.388,1030.752,1642.091,1031.301,1642.780,1031.867C1643.469,1032.433,1644.144,1033.015,1644.805,1033.614C1645.466,1034.213,1646.112,1034.828,1646.743,1035.458C1647.374,1036.089,1647.988,1036.735,1648.587,1037.396C1649.186,1038.057,1649.769,1038.732,1650.334,1039.421C1650.900,1040.111,1651.449,1040.814,1651.980,1041.530C1652.511,1042.246,1653.025,1042.975,1653.520,1043.717C1654.016,1044.459,1654.493,1045.212,1654.951,1045.977C1655.410,1046.742,1655.849,1047.518,1656.270,1048.304C1656.690,1049.091,1657.091,1049.887,1657.473,1050.694C1657.854,1051.500,1658.216,1052.315,1658.557,1053.139C1658.898,1053.963,1659.218,1054.795,1659.519,1055.635C1659.820,1056.475,1660.099,1057.321,1660.358,1058.175C1660.617,1059.028,1660.855,1059.888,1661.072,1060.753C1661.289,1061.618,1661.484,1062.488,1661.658,1063.363C1661.832,1064.238,1661.984,1065.116,1662.115,1065.998C1662.246,1066.881,1662.355,1067.766,1662.443,1068.653C1662.530,1069.541,1662.596,1070.430,1662.639,1071.321C1662.683,1072.212,1662.705,1073.103,1662.705,1073.995L1662.705,1096.997C1662.705,1097.889,1662.683,1098.780,1662.639,1099.671C1662.596,1100.562,1662.530,1101.451,1662.443,1102.339C1662.355,1103.226,1662.246,1104.111,1662.115,1104.994C1661.984,1105.876,1661.832,1106.754,1661.658,1107.629C1661.484,1108.504,1661.289,1109.374,1661.072,1110.239C1660.855,1111.104,1660.617,1111.964,1660.358,1112.817C1660.099,1113.671,1659.820,1114.518,1659.519,1115.357C1659.218,1116.197,1658.897,1117.029,1658.556,1117.853C1658.215,1118.677,1657.853,1119.492,1657.472,1120.298C1657.091,1121.105,1656.690,1121.901,1656.270,1122.688C1655.849,1123.475,1655.410,1124.250,1654.951,1125.015C1654.493,1125.780,1654.016,1126.534,1653.520,1127.275C1653.025,1128.016,1652.511,1128.745,1651.980,1129.462C1651.449,1130.179,1650.900,1130.882,1650.334,1131.571C1649.769,1132.260,1649.186,1132.935,1648.587,1133.596C1647.988,1134.257,1647.374,1134.903,1646.743,1135.534C1646.112,1136.165,1645.466,1136.779,1644.805,1137.378C1644.144,1137.977,1643.469,1138.560,1642.780,1139.125C1642.091,1139.691,1641.388,1140.240,1640.671,1140.771C1639.954,1141.302,1639.225,1141.816,1638.484,1142.311C1637.743,1142.807,1636.989,1143.284,1636.224,1143.742C1635.459,1144.201,1634.684,1144.640,1633.897,1145.061C1633.110,1145.481,1632.314,1145.882,1631.507,1146.264C1630.701,1146.645,1629.886,1147.007,1629.062,1147.348C1628.238,1147.689,1627.406,1148.009,1626.566,1148.310C1625.726,1148.611,1624.879,1148.890,1624.026,1149.149C1623.173,1149.408,1622.313,1149.646,1621.448,1149.863C1620.583,1150.080,1619.713,1150.275,1618.838,1150.449C1617.963,1150.623,1617.085,1150.775,1616.203,1150.906C1615.320,1151.037,1614.435,1151.146,1613.548,1151.234C1612.660,1151.321,1611.771,1151.387,1610.880,1151.430C1609.989,1151.474,1609.098,1151.496,1608.206,1151.496L1607.204,1151.496C1606.312,1151.496,1605.421,1151.474,1604.530,1151.430C1603.639,1151.387,1602.750,1151.321,1601.862,1151.234C1600.975,1151.146,1600.090,1151.037,1599.207,1150.906C1598.325,1150.775,1597.447,1150.623,1596.572,1150.449C1595.697,1150.275,1594.827,1150.080,1593.962,1149.863C1593.097,1149.646,1592.237,1149.408,1591.384,1149.149C1590.530,1148.890,1589.683,1148.611,1588.844,1148.310C1588.004,1148.009,1587.172,1147.689,1586.348,1147.348C1585.524,1147.007,1584.709,1146.645,1583.903,1146.264C1583.096,1145.882,1582.300,1145.481,1581.513,1145.061C1580.726,1144.640,1579.951,1144.201,1579.186,1143.742C1578.421,1143.284,1577.667,1142.807,1576.926,1142.311C1576.185,1141.816,1575.456,1141.302,1574.739,1140.771C1574.023,1140.240,1573.320,1139.691,1572.630,1139.125C1571.941,1138.560,1571.266,1137.977,1570.605,1137.378C1569.944,1136.779,1569.298,1136.165,1568.667,1135.534C1568.037,1134.903,1567.422,1134.257,1566.823,1133.596C1566.224,1132.935,1565.641,1132.260,1565.076,1131.571C1564.510,1130.882,1563.961,1130.179,1563.430,1129.462C1562.899,1128.745,1562.385,1128.016,1561.890,1127.275C1561.394,1126.534,1560.917,1125.780,1560.459,1125.015C1560.000,1124.250,1559.561,1123.475,1559.140,1122.688C1558.720,1121.901,1558.319,1121.105,1557.937,1120.298C1557.556,1119.492,1557.195,1118.677,1556.853,1117.853C1556.512,1117.029,1556.191,1116.197,1555.891,1115.357C1555.590,1114.518,1555.311,1113.671,1555.052,1112.817C1554.793,1111.964,1554.555,1111.104,1554.338,1110.239C1554.122,1109.374,1553.926,1108.504,1553.752,1107.629C1553.578,1106.754,1553.426,1105.876,1553.295,1104.994C1553.164,1104.111,1553.055,1103.226,1552.967,1102.339C1552.880,1101.451,1552.814,1100.562,1552.771,1099.671C1552.727,1098.780,1552.705,1097.889,1552.705,1096.997L1552.705,1073.995C1552.705,1073.103,1552.727,1072.212,1552.771,1071.321C1552.814,1070.430,1552.880,1069.541,1552.967,1068.653C1553.055,1067.766,1553.164,1066.881,1553.295,1065.998C1553.426,1065.116,1553.578,1064.238,1553.752,1063.363C1553.926,1062.488,1554.122,1061.618,1554.338,1060.753C1554.555,1059.888,1554.793,1059.028,1555.052,1058.175C1555.311,1057.321,1555.590,1056.475,1555.891,1055.635C1556.191,1054.795,1556.512,1053.963,1556.853,1053.139C1557.195,1052.315,1557.556,1051.500,1557.937,1050.694C1558.319,1049.887,1558.720,1049.091,1559.140,1048.304C1559.561,1047.518,1560.000,1046.742,1560.459,1045.977C1560.917,1045.212,1561.394,1044.459,1561.890,1043.717C1562.385,1042.975,1562.899,1042.246,1563.430,1041.530C1563.961,1040.814,1564.510,1040.111,1565.076,1039.421C1565.641,1038.732,1566.224,1038.057,1566.823,1037.396C1567.422,1036.735,1568.037,1036.089,1568.667,1035.458C1569.298,1034.828,1569.944,1034.213,1570.605,1033.614C1571.266,1033.015,1571.941,1032.433,1572.630,1031.867C1573.320,1031.301,1574.023,1030.752,1574.739,1030.221C1575.456,1029.690,1576.185,1029.176,1576.926,1028.681C1577.667,1028.185,1578.421,1027.708,1579.186,1027.250C1579.951,1026.791,1580.726,1026.352,1581.513,1025.931C1582.300,1025.511,1583.096,1025.110,1583.903,1024.729C1584.709,1024.347,1585.524,1023.986,1586.348,1023.645C1587.172,1023.303,1588.004,1022.982,1588.844,1022.682C1589.683,1022.381,1590.530,1022.102,1591.384,1021.843C1592.237,1021.584,1593.097,1021.346,1593.962,1021.129C1594.827,1020.913,1595.697,1020.717,1596.572,1020.543C1597.447,1020.369,1598.325,1020.217,1599.207,1020.086C1600.090,1019.955,1600.975,1019.846,1601.862,1019.758C1602.750,1019.671,1603.639,1019.605,1604.530,1019.562C1605.421,1019.518,1606.312,1019.496,1607.204,1019.496ZZZZZZZ" fill-rule="evenodd" style="fill: rgb(59, 107, 173); fill-opacity: 1;" class="fills"/></svg>
                    "##;

                let canvas = self.drawing_surface.canvas();

                let font_mgr = skia::FontMgr::new();
                let typeface = font_mgr
                     .new_from_data(ROBOTO_REGULAR, None)
                     .expect("Failed to load ROBOTO font");

                let typeface_font_provider = {
                    let mut typeface_font_provider = skia::textlayout::TypefaceFontProvider::new();
                    // We need a system font manager to be able to load typefaces.
                    let font_mgr = skia::FontMgr::new();
                    let typeface = font_mgr
                        .new_from_data(ROBOTO_REGULAR, None)
                        .expect("Failed to load Ubuntu font");

                    typeface_font_provider.register_typeface(typeface, TYPEFACE_ALIAS);
                    typeface_font_provider
                };

                let mut font_collection = skia::textlayout::FontCollection::new();
                font_collection.set_default_font_manager(Some(typeface_font_provider.into()), None);
                let font_mgr_2 = font_collection.fallback_manager().unwrap();
                let dom = skia::svg::Dom::from_str(svg, font_mgr_2).unwrap();
                dom.render(canvas);

            }
            (_, Kind::Path(path)) => {
                self.drawing_surface
                    .canvas()
                    .draw_path(&path.to_skia_path(), &fill.to_paint(&selrect));
            }
        }
    }

    fn render_all_from_cache(&mut self) -> Result<(), String> {
        self.reset_canvas();

        let cached = self
            .cached_surface_image
            .as_ref()
            .ok_or("Uninitialized cached surface image")?;

        let image = &cached.image;
        let paint = skia::Paint::default();
        self.final_surface.canvas().save();
        self.drawing_surface.canvas().save();

        let navigate_zoom = self.viewbox.zoom / cached.viewbox.zoom;
        let navigate_x = cached.viewbox.zoom * (self.viewbox.pan_x - cached.viewbox.pan_x);
        let navigate_y = cached.viewbox.zoom * (self.viewbox.pan_y - cached.viewbox.pan_y);

        self.final_surface
            .canvas()
            .scale((navigate_zoom, navigate_zoom));
        self.final_surface.canvas().translate((
            navigate_x * self.options.dpr(),
            navigate_y * self.options.dpr(),
        ));
        self.final_surface
            .canvas()
            .draw_image(image.clone(), (0, 0), Some(&paint));

        self.final_surface.canvas().restore();
        self.drawing_surface.canvas().restore();

        self.flush();

        Ok(())
    }

    fn render_debug_view(&mut self) {
        let mut paint = skia::Paint::default();
        paint.set_style(skia::PaintStyle::Stroke);
        paint.set_color(skia::Color::from_argb(255, 255, 0, 255));
        paint.set_stroke_width(1.);

        let mut scaled_rect = self.viewbox.area.clone();
        let x = 100. + scaled_rect.x() * 0.2;
        let y = 100. + scaled_rect.y() * 0.2;
        let width = scaled_rect.width() * 0.2;
        let height = scaled_rect.height() * 0.2;
        scaled_rect.set_xywh(x, y, width, height);

        self.debug_surface.canvas().draw_rect(scaled_rect, &paint);
    }

    fn render_debug_shape(&mut self, shape: &Shape, intersected: bool) {
        let mut paint = skia::Paint::default();
        paint.set_style(skia::PaintStyle::Stroke);
        paint.set_color(if intersected {
            skia::Color::from_argb(255, 255, 255, 0)
        } else {
            skia::Color::from_argb(255, 0, 255, 255)
        });
        paint.set_stroke_width(1.);

        let mut scaled_rect = shape.selrect.clone();
        let x = 100. + scaled_rect.x() * 0.2;
        let y = 100. + scaled_rect.y() * 0.2;
        let width = scaled_rect.width() * 0.2;
        let height = scaled_rect.height() * 0.2;
        scaled_rect.set_xywh(x, y, width, height);

        self.debug_surface.canvas().draw_rect(scaled_rect, &paint);
    }

    fn render_debug(&mut self) {
        let paint = skia::Paint::default();
        self.render_debug_view();
        self.debug_surface.draw(
            &mut self.final_surface.canvas(),
            (0.0, 0.0),
            skia::SamplingOptions::new(skia::FilterMode::Linear, skia::MipmapMode::Nearest),
            Some(&paint),
        );
    }

    // Returns a boolean indicating if the viewbox contains the rendered shapes
    fn render_shape_tree(&mut self, id: &Uuid, shapes: &HashMap<Uuid, Shape>) -> bool {
        let shape = shapes.get(&id).unwrap();
        let mut is_complete = self.viewbox.area.contains(shape.selrect);

        if !id.is_nil() {
            if !shape.selrect.intersects(self.viewbox.area) || shape.hidden {
                self.render_debug_shape(shape, false);
                // TODO: This means that not all the shapes are renderer so we
                // need to call a render_all on the zoom out.
                return is_complete; // TODO return is_complete or return false??
            } else {
                self.render_debug_shape(shape, true);
            }
        }

        // This is needed so the next non-children shape does not carry this shape's transform
        self.final_surface.canvas().save();
        self.drawing_surface.canvas().save();

        if !id.is_nil() {
            self.render_single_shape(shape);
        }

        // draw all the children shapes
        let shape_ids = shape.children.iter();
        for shape_id in shape_ids {
            is_complete = self.render_shape_tree(shape_id, shapes) && is_complete;
        }

        self.final_surface.canvas().restore();
        self.drawing_surface.canvas().restore();
        return is_complete;
    }
}
