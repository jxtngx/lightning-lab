import { Container, Grid } from "@mui/material";

import { LabelDropdown } from "./labelDropdown";
import { ModelCard } from "./modelCard";

export const SideBarGrid = () => (
    <Grid container lg={3} sm={6} xl={3} xs={12} direction={"column"}>
        <Grid>
            <LabelDropdown />
        </Grid>
        <br />
        <Grid>
            <ModelCard />
        </Grid>
    </Grid>
);
